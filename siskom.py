import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import folium
from streamlit_folium import st_folium

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

# Fungsi untuk menghitung Cosine Similarity antar tempat wisata berdasarkan deskripsi
def calculate_cosine_similarity_between_places(data, place_id):
    # Ambil deskripsi tempat wisata yang dipilih oleh pengguna
    selected_place_desc = data[data['Place_Id'] == place_id]['Description'].values[0]

    # Gunakan TfidfVectorizer untuk representasi TF-IDF dari deskripsi tempat wisata
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['Description'])

    # Hitung Cosine Similarity antar tempat wisata dalam dataset
    cosine_sim = cosine_similarity(tfidf_matrix)

    # Ambil nilai Cosine Similarity untuk tempat wisata yang dipilih
    selected_place_index = data[data['Place_Id'] == place_id].index[0]
    sim_scores = cosine_sim[selected_place_index]

    # Buat DataFrame untuk menampilkan skor similarity dan informasi tempat wisata
    sim_scores_df = pd.DataFrame(sim_scores, columns=['Similarity_Score'])
    sim_scores_df['Place_Id'] = data['Place_Id']
    sim_scores_df['Place_Name'] = data['Place_Name']
    sim_scores_df['Category'] = data['Category']
    sim_scores_df['Price_Display'] = data['Price_Display']
    sim_scores_df['Rating'] = data['Rating']
    
    # Urutkan berdasarkan similarity score tertinggi
    sim_scores_df = sim_scores_df.sort_values(by='Similarity_Score', ascending=False)

    # Tampilkan 5 tempat wisata teratas berdasarkan similarity
    top_n_recommendations = sim_scores_df.iloc[1:6]
    return top_n_recommendations

# Muat dataset
data = load_data_from_drive()

# Periksa jika data kosong
if data.empty:
    st.error("Dataset kosong atau tidak valid. Pastikan dataset memiliki data yang diperlukan.")
    st.stop()

# Format harga setelah normalisasi
data['Price'] = data['Price'].replace({'Rp ': '', ',': ''}, regex=True).astype(int)
data['Price_Display'] = data['Price'].apply(lambda x: format_rupiah(x))

# Streamlit Interface
st.set_page_config(page_title="Sistem Rekomendasi Tempat Wisata Yogyakarta", layout="wide")

# Header dan Deskripsi Aplikasi
st.markdown(
    """
    <h1 style="text-align: center; color: #0072bb;">Sistem Rekomendasi Tempat Wisata Yogyakarta</h1>
    <p style="text-align: center; font-size: 18px;">Temukan tempat wisata terbaik di Yogyakarta berdasarkan preferensi Anda.</p>
    """, unsafe_allow_html=True)

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
    st.write(selected_place['Description'])

# Pilih jumlah rekomendasi
recommendation_count = st.selectbox("Pilih jumlah rekomendasi:", [1, 3, 5])

# Rekomendasi tempat wisata berdasarkan deskripsi yang dipilih
st.subheader(f"Rekomendasi {recommendation_count} Tempat Wisata Serupa")
recommendations = calculate_cosine_similarity_between_places(data, place_id)

# Pilih rekomendasi untuk melihat detail
selected_recommendation_name = st.selectbox("Pilih tempat wisata untuk melihat detail:", recommendations['Place_Name'])

# Menampilkan detail rekomendasi jika dipilih
if selected_recommendation_name:
    recommended_place = recommendations[recommendations['Place_Name'] == selected_recommendation_name].iloc[0]
    st.write(f"### Detail: {recommended_place['Place_Name']}")
    st.write(f"**Kategori:** {recommended_place['Category']}")
    st.write(f"**Harga:** {recommended_place['Price_Display']}")
    st.write(f"**Rating:** {recommended_place['Rating']}")
    st.write(f"**Skor Similarity:** {recommended_place['Similarity_Score']:.4f}")
    st.write("---")

# Persiapkan data untuk peta
if 'Latitude' in data.columns and 'Longitude' in data.columns:
    data = data.rename(columns={'Latitude': 'lat', 'Longitude': 'lon'})

    # Membuat peta dengan folium
    m = folium.Map(location=[data['lat'].mean(), data['lon'].mean()], zoom_start=12)

    # Menambahkan marker untuk setiap tempat
    for idx, row in data.iterrows():
        place_name_cleaned = row['Place_Name']
        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=folium.Popup(
                f"<b>{place_name_cleaned}</b><br>Harga: {row['Price_Display']}<br>Rating: {row['Rating']}<br>Koordinat: ({row['lat']}, {row['lon']})", 
                max_width=300
            ),
            icon=folium.Icon(color="blue", icon="info-sign")  # Setiap marker menggunakan ikon khusus
        ).add_to(m)

    # Fokuskan peta ke tempat yang dipilih
    if 'Latitude' in selected_place and 'Longitude' in selected_place:
        folium.Marker(
            location=[selected_place['Latitude'], selected_place['Longitude']],
            popup=folium.Popup(
                f"<b>{selected_place['Place_Name']}</b><br>Harga: {selected_place['Price_Display']}<br>Rating: {selected_place['Rating']}<br>Koordinat: ({selected_place['Latitude']}, {selected_place['Longitude']})",
                max_width=300
            ),
            icon=folium.Icon(color="red", icon="info-sign")  # Ikon merah untuk tempat yang dipilih
        ).add_to(m)

        # Menyelaraskan peta dengan lokasi tempat yang dipilih
        m.location = [selected_place['Latitude'], selected_place['Longitude']]
        m.zoom_start = 14  # Fokus lebih dekat pada tempat yang dipilih

    # Menampilkan peta di Streamlit
    st.subheader("Peta Lokasi Tempat Wisata")
    st_folium(m, width=700)
else:
    st.warning("Dataset tidak memiliki kolom Latitude dan Longitude. Peta tidak dapat ditampilkan.")
