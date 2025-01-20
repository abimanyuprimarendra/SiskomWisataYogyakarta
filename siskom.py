import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import folium
from streamlit_folium import st_folium
import re

# Fungsi untuk memuat data dari Google Drive
@st.cache_data
def load_data_from_drive():
    try:
        csv_url = "https://drive.google.com/uc?id=1ChhGYhX2IktjQQRjQQpLzewYPIpOEI0g"  # URL yang sudah benar
        data = pd.read_csv(csv_url)
        return data
    except Exception as e:
        st.error(f"Gagal memuat dataset. Pesan kesalahan: {str(e)}")
        return pd.DataFrame()

# Fungsi untuk memformat harga menjadi format Rupiah
def format_rupiah(angka):
    return "Rp {:,.0f}".format(angka).replace(",", ".")

# Fungsi untuk menghilangkan tanda baca dari teks
def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

# Muat dataset
data = load_data_from_drive()

# Periksa jika data kosong
if data.empty:
    st.error("Dataset kosong atau tidak valid. Pastikan dataset memiliki data yang diperlukan.")
    st.stop()

# Hilangkan kolom Coordinate jika tidak dibutuhkan
data = data.drop(columns=['Coordinate'], errors='ignore')

# Preprocessing kolom Description
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['Description'])

# Mengonversi Price menjadi numerik untuk perhitungan
data['Price'] = data['Price'].replace({'Rp ': '', ',': ''}, regex=True).astype(int)

# Format harga setelah normalisasi
data['Price_Display'] = data['Price'].apply(lambda x: format_rupiah(x))

# Perhitungan similarity
description_sim = cosine_similarity(tfidf_matrix)
price_sim = cosine_similarity(data[['Price']].values.reshape(-1, 1))
rating_sim = cosine_similarity(data[['Rating']].values.reshape(-1, 1))

# Gabungkan similarity tanpa bobot
description_weight = 1
price_weight = 1
rating_weight = 1

final_similarity = (description_weight * description_sim + 
                    price_weight * price_sim + 
                    rating_weight * rating_sim)

# Fungsi untuk merekomendasikan tempat
def recommend(place_id, top_n=5):
    try:
        idx = data[data['Place_Id'] == place_id].index[0]
    except IndexError:
        st.error(f"Place ID {place_id} tidak ditemukan dalam data.")
        st.stop()

    sim_scores = pd.DataFrame(final_similarity[idx], columns=['Score'])
    sim_scores['Place_Id'] = data['Place_Id']
    sim_scores = sim_scores.sort_values(by='Score', ascending=False).iloc[1:top_n+1]
    recommended_data = data[data['Place_Id'].isin(sim_scores['Place_Id'])][['Place_Id', 'Place_Name', 'Category', 'Rating', 'Price_Display']]
    recommended_data = recommended_data.merge(sim_scores, on='Place_Id')
    return recommended_data

# Streamlit Interface
st.title("Sistem Rekomendasi Tempat Wisata Yogyakarta")

# Filter berdasarkan kategori
categories = data['Category'].unique()
selected_category = st.selectbox("Pilih kategori:", ["Semua"] + list(categories))

if selected_category != "Semua":
    data = data[data['Category'] == selected_category]

# Input pengguna
place_name = st.selectbox("Pilih tempat wisata:", data['Place_Name'])
selected_place = data[data['Place_Name'] == place_name].iloc[0]
place_id = selected_place['Place_Id']

# Tata letak informasi wisata
st.subheader("Informasi Tempat Wisata")
col1, col2 = st.columns([1, 2])
with col1:
    st.write("**Nama:**")
    st.write("**Kategori:**")
    st.write("**Harga:**")
    st.write("**Rating:**")
    st.write("**Deskripsi:**")
with col2:
    st.write(selected_place['Place_Name'])
    st.write(selected_place['Category'])
    st.write(selected_place['Price_Display'])
    st.write(selected_place['Rating'])
    st.write(remove_punctuation(selected_place['Description']))

# Rekomendasi tempat
st.subheader("Rekomendasi Tempat Wisata Serupa")
top_n_recommendations = 5  # Jumlah rekomendasi yang ingin ditampilkan
st.write(f"Rekomendasi tempat wisata serupa untuk {selected_place['Place_Name']}:")

recommendations = recommend(place_id, top_n_recommendations)

# Pilihan opsi untuk rekomendasi
selected_recommendation = st.selectbox(
    "Pilih rekomendasi untuk melihat detail:",
    recommendations['Place_Name']
)

# Tampilkan detail rekomendasi jika dipilih
if selected_recommendation:
    recommended_place = recommendations[recommendations['Place_Name'] == selected_recommendation].iloc[0]
    st.write(f"### Detail: {recommended_place['Place_Name']}")
    st.write(f"**Kategori:** {recommended_place['Category']}")
    st.write(f"**Harga:** {recommended_place['Price_Display']}")
    st.write(f"**Rating:** {recommended_place['Rating']}")
    st.write(f"**Total Similarity:** {recommended_place['Score']:.4f}")


# Persiapkan data untuk peta
if 'Latitude' in data.columns and 'Longitude' in data.columns:
    data = data.rename(columns={'Latitude': 'lat', 'Longitude': 'lon'})

    # Membuat peta dengan folium
    m = folium.Map(location=[data['lat'].mean(), data['lon'].mean()], zoom_start=12)

    # Menambahkan marker untuk setiap tempat
    for idx, row in data.iterrows():
        place_name_cleaned = remove_punctuation(row['Place_Name'])
        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=folium.Popup(
                f"<b>{place_name_cleaned}</b><br>Harga: {row['Price_Display']}<br>Rating: {row['Rating']}<br>Koordinat: ({row['lat']}, {row['lon']})", 
                max_width=300
            )
        ).add_to(m)

    # Fokuskan peta ke tempat yang dipilih
    if 'Latitude' in selected_place and 'Longitude' in selected_place:
        folium.Marker(
            location=[selected_place['Latitude'], selected_place['Longitude']],
            popup=folium.Popup(
                f"<b>{selected_place['Place_Name']}</b><br>Harga: {selected_place['Price_Display']}<br>Rating: {selected_place['Rating']}<br>Koordinat: ({selected_place['Latitude']}, {selected_place['Longitude']})",
                max_width=300
            ),
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(m)

        # Menyelaraskan peta dengan lokasi tempat yang dipilih
        m.location = [selected_place['Latitude'], selected_place['Longitude']]
        m.zoom_start = 14  # Fokus lebih dekat pada tempat yang dipilih

    # Menampilkan peta di Streamlit
    st.subheader("Peta Lokasi Tempat Wisata")
    st_folium(m, width=700)
else:
    st.warning("Dataset tidak memiliki kolom Latitude dan Longitude. Peta tidak dapat ditampilkan.")
