import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gspread
from google.oauth2.service_account import Credentials

# ==========================================
# 1. KONFIGURASI TAMPILAN (CSS CUSTOM)
# ==========================================
st.set_page_config(page_title="Rekomendasi Divisi Pramuka", layout="wide")

# CSS untuk tampilan yang lebih premium dan bersih
st.markdown("""
<style>
    /* Background utama */
    .main {
        background-color: #f0f2f6;
    }
    /* Judul Utama */
    h1 {
        color: #1E5128;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 700;
        padding-bottom: 10px;
        border-bottom: 3px solid #1E5128;
        margin-bottom: 20px;
    }
    /* Sub Judul Form */
    h3 {
        color: #1E5128;
    }
    /* Tombol Proses */
    .stButton>button {
        background-color: #1E5128;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 10px 24px;
        border-radius: 8px;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #3E7C17;
        border-color: #3E7C17;
    }
    /* Box Hasil */
    .result-box {
        background-color: #D8E9A8;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1E5128;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    /* Sembunyikan Footer Streamlit */
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. KONEKSI & LOAD MODEL
# ==========================================
def init_connection():
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    
    # Cek apakah kita sedang berjalan di Streamlit Cloud (menggunakan Secrets)
    if "gcp_service_account" in st.secrets:
        # Buat kredensial dari Secrets
        creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
    else:
        # Jika di komputer lokal, pakai file json
        try:
            creds = Credentials.from_service_account_file("credentials.json", scopes=scopes)
        except FileNotFoundError:
            return None
            
    client = gspread.authorize(creds)
    return client

@st.cache_resource
def load_model():
    try:
        with open("model_artifacts.pkl", "rb") as f:
            return pickle.load(f)
    except:
        return None

artifacts = load_model()
if artifacts is None:
    st.error("File model tidak ditemukan. Jalankan Training Notebook terlebih dahulu.")
    st.stop()

model = artifacts['model']
encoders = artifacts['encoders']
fitur_kolom = artifacts['fitur_kolom']
target_col = artifacts['target_col']

st.title("Sistem Rekomendasi Divisi Organisasi")
st.markdown("""
Selamat datang di sistem penentuan divisi berbasis **Machine Learning**. 
Sistem ini akan menganalisis minat dan bakat Anda untuk merekomendasikan divisi yang paling tepat.
""")

with st.expander("Petunjuk Pengisian"):
    st.markdown("""
    1. Isi **Nama Lengkap** dan **Kelas** dengan benar.
    2. Status otomatis terisi **Calon Dewan**.
    3. Jawab pertanyaan kuesioner pada skala 1 - 5.
    4. Tekan tombol **Proses Rekomendasi** di bawah.
    """)

st.markdown("---")

with st.form("form_rekomendasi"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        nama = st.text_input("Nama Lengkap", placeholder="Masukkan nama lengkap...")
    with col2:
        kelas = st.text_input("Kelas", placeholder="Contoh: X.1")
    with col3:
        st.text_input("Status", value="Calon Dewan", disabled=True)

    st.markdown("### Kuesioner Minat & Bakat")
    st.caption("Skala 1 (Sangat Tidak Setuju) hingga 5 (Sangat Setuju)")

    col_kiri, col_kanan = st.columns(2)
    
    input_user = {}
    index = 0
    
    for col in fitur_kolom:
        target_form = col_kiri if index % 2 == 0 else col_kanan
        
        with target_form:
            if col == 'Status':
                input_user[col] = "Calon Dewan"            
            elif col in encoders:
                options = encoders[col].classes_.tolist()
                input_user[col] = st.selectbox(f"{col}", options)            
            else:
                input_user[col] = st.slider(f"{col}", 1, 5, 3)
        index += 1
    st.markdown("") 
    submitted = st.form_submit_button("PROSES REKOMENDASI")

if submitted:
    if not nama or not kelas:
        st.warning("Nama dan Kelas wajib diisi.")
    else:
        df_input = pd.DataFrame([input_user])
        df_input = df_input[fitur_kolom]

        for col in df_input.columns:
            if col in encoders:
                try:
                    df_input[col] = encoders[col].transform(df_input[col])
                except:
                    df_input[col] = 0

        pred = model.predict(df_input)[0]
        hasil_divisi = encoders[target_col].inverse_transform([pred])[0]

        st.markdown("---")
        st.markdown(f"""
        <div class="result-box">
            <h3 style="text-align:center; margin-bottom:5px;">Rekomendasi Divisi</h3>
            <h1 style="text-align:center; color:#1E5128; margin-top:0px;">{hasil_divisi}</h1>
            <p style="text-align:center;">{nama} ({kelas}) direkomendasikan untuk bergabung di divisi ini.</p>
        </div>
        """, unsafe_allow_html=True)
        try:
            client = init_connection()
            if client:
                SPREADSHEET_ID = '1DS2XgPwtqnCV7wOAumq02X4IdbkiZ5abS5df2x28D88'
                sheet = client.open_by_key(SPREADSHEET_ID).sheet1                
                row_data = [nama, kelas]                
                for col in fitur_kolom:
                    row_data.append(input_user[col])
                row_data.append(hasil_divisi)
                sheet.append_row(row_data)
                st.success("Terima kasih! Data Anda telah berhasil tercatat.")
            else:
                st.error("Gagal koneksi ke Google Cloud. Cek file credentials.json.")
        
        except gspread.exceptions.SpreadsheetNotFound:
            st.error("Spreadsheet tidak ditemukan. Pastikan ID sudah benar.")
        except Exception as e:
            st.error(f"Terjadi error: {e}")