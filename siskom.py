import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

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

# Rekomendasi tempat wisata berdasarkan deskripsi yang dipilih
st.subheader("Rekomendasi Tempat Wisata Serupa")
recommendations = calculate_cosine_similarity_between_places(data, place_id)

# Menampilkan rekomendasi tempat wisata
for index, recommendation in recommendations.iterrows():
    st.write(f"**Nama Tempat:** {recommendation['Place_Name']}")
    st.write(f"**Kategori:** {recommendation['Category']}")
    st.write(f"**Harga:** {recommendation['Price_Display']}")
    st.write(f"**Rating:** {recommendation['Rating']}")
    st.write(f"**Skor Similarity:** {recommendation['Similarity_Score']:.4f}")
    st.write("---")
