# Sistem Rekomendasi Divisi Pramuka

## 1) Gambaran Umum

Proyek ini adalah aplikasi rekomendasi berbasis machine learning untuk menentukan divisi yang paling cocok untuk seorang siswa Pramuka berdasarkan hasil kuesioner. Aplikasi terdiri dari dua bagian utama:

1. Pipeline pelatihan model (`train_model.py`) untuk mempersiapkan data, membandingkan beberapa algoritme, dan menyimpan artifact model.
2. Aplikasi web interaktif dengan Streamlit (`app.py`) untuk prediksi, dashboard evaluasi, dan prediksi batch.

Tujuan proyek:

- Mengotomatisasi rekomendasi divisi secara konsisten.
- Menyediakan rekomendasi yang dapat dijelaskan (probabilitas + aturan logika pembanding).
- Menyediakan dashboard evaluasi yang jelas untuk transparansi model.

## 2) Struktur Proyek

- `train_model.py`
  - Membaca dataset.
  - Membersihkan dan memvalidasi fitur.
  - Membangun `ColumnTransformer` + `Pipeline`.
  - Membandingkan minimal 3 model (RandomForest, ExtraTrees, Logistic Regression) dengan `StratifiedKFold`.
  - Menyimpan model terbaik ke `model_artifacts.pkl`.

- `app.py`
  - Memuat artifact model.
  - Menyediakan form input peserta.
  - Menampilkan dashboard model (akurasi, precision, recall, F1, confusion matrix, distribusi kelas).
  - Menyediakan rekomendasi Top-3 probabilitas.
  - Menyediakan perbandingan skor manual/logic sebagai penjelas.
  - Menyediakan upload batch CSV/Excel dan ekspor hasil prediksi.
  - Integrasi Google Sheets optional (tidak menghambat jika kredensial tidak tersedia).

- `model_artifacts.pkl`
  - Hasil pelatihan terbaru yang memuat:
    - pipeline model terbaik
    - kolom fitur
    - pembagian fitur numerik dan kategorikal
    - nama kelas
    - metrik evaluasi
    - schema dataset
    - timestamp pembuatan

- `requirements.txt`
  - Dependensi Python yang diperlukan.

- `.gitignore`
  - Menyimpan direktori/berkas sensitif agar tidak masuk versi kontrol.

## 3) Instalasi

Pastikan Python 3.10+ terpasang.

```bash
pip install -r requirements.txt
```

## 4) Dataset

Pipeline default menggunakan dataset sintetik:

- `datasheet_rekomendasi_divisi_pramuka_500.xlsx`

Dataset wajib memiliki kolom target berikut:

- `Divisi yang paling di minati?`

Kolom target **dilarang** digunakan sebagai fitur pada prediksi baru; hanya untuk training dan evaluasi.

Kolom yang tidak boleh dipakai sebagai fitur karena non-fitur identitas:

- `Nama Lengkap`
- `Kelas`
- `HEAD` (jika ada)

Kolom pengalaman organisasi:

- `Pernah ikut organisasi Sebelumnya?`

diperlakukan sebagai skala 1-5 (bukan YA/TIDAK).

Catatan:

- Data yang digunakan untuk proyek ini bersifat **sintetis** untuk keperluan prototype.

## 5) Cara Melatih Ulang Model

Jalankan:

```bash
python train_model.py
```

Atau tentukan path dataset secara eksplisit:

```bash
python train_model.py --data "C:\\path\\ke\\datasheet_rekomendasi_divisi_pramuka_500.xlsx"
```

Script akan:

1. Memuat dataset.
2. Memvalidasi target dan fitur.
3. Menyiapkan pipeline preprocessing.
4. Mengevaluasi kandidat model dengan `StratifiedKFold`.
5. Menyimpan artifact ke `model_artifacts.pkl`.

## 6) Menjalankan Aplikasi Streamlit

```bash
streamlit run app.py
```

Aplikasi akan:

- Memuat `model_artifacts.pkl`.
- Menampilkan form prediksi individu.
- Memproses rekomendasi batch (CSV/XLSX).
- Menampilkan dashboard evaluasi model.

## 7) Spesifikasi Input di Aplikasi

- Nama dan kelas **tetap diinput manual**.
- Status peserta otomatis bernilai **`Calon Dewan`** (tidak perlu dipilih manual).
- Input numerik (Likert): slider 1–5.
- Input kategorikal: selectbox sesuai domain data training.

Validasi dilakukan untuk:

- Kolom wajib.
- Rentang skala Likert 1–5.
- Kategori sesuai pilihan yang valid.
- Format file upload CSV/XLSX.

## 8) Fitur Dashboard

- Distribusi data per divisi.
- Metrik akurasi, precision macro, recall macro, F1 macro.
- Confusion matrix.
- Jumlah data training dan jumlah fitur.
- Peringatan jika ada kelas dengan jumlah data < 50.

## 9) Variabel Konfigurasi (Opsional)

Integrasi Google Sheets dan path dataset dapat disetel lewat `.env` / `st.secrets`.

Contoh `.env`:

```env
PRAMUKA_DATA_PATH=C:\\Users\\ASUS\\Downloads\\datasheet_rekomendasi_divisi_pramuka_500.xlsx
GOOGLE_SHEET_ID=your_sheet_id_here
```

Jika kredensial Google tidak tersedia, aplikasi tetap berjalan normal tanpa fitur penyimpanan ke Sheets.

## 10) Kontribusi

1. Update dataset (format tetap menjaga kolom target).
2. Jalankan ulang training.
3. Commit hasil perubahan.
4. Jalankan aplikasi untuk verifikasi.
