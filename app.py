import streamlit as st
import streamlit.components.v1 as stc
import pandas as pd
import numpy as np
import pickle
import datetime
import os

# ===================================================================================
# Konfigurasi Halaman Streamlit
# ===================================================================================
st.set_page_config(
    page_title="Prediksi Sewa Sepeda COGNIDATA",
    page_icon="🚲",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'mailto:fauzanzahid720@gmail.com', # Pastikan email ini benar
        'Report a bug': "mailto:fauzanzahid720@gmail.com",
        'About': "### Aplikasi Prediksi Permintaan Sepeda\nTim COGNIDATA\nPowered by XGBoost & Scikit-learn."
    }
)

# ===================================================================================
# Muat Model
# ===================================================================================
@st.cache_resource # Menggunakan cache_resource untuk objek model
def load_pickled_model(model_path):
    """Memuat model dari file pickle."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        print(f"Model berhasil dimuat dari: {model_path}")
        return model
    except FileNotFoundError:
        st.error(f"File model '{model_path}' tidak ditemukan. Pastikan file ada di direktori yang sama dengan aplikasi atau path-nya benar.")
        return None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
        return None

# PERHATIKAN: Nama file model disesuaikan dengan output notebook Colab terakhir
MODEL_FILENAME = 'XGBoost_SKLearn_Pipeline_Final.pkl' 
pipeline_model = load_pickled_model(MODEL_FILENAME)

# ===================================================================================
# HTML Templates
# ===================================================================================
PRIMARY_BG_COLOR = "#003366"
PRIMARY_TEXT_COLOR = "#FFFFFF"
ACCENT_COLOR = "#FFD700"

HTML_BANNER = f"""
    <div style="background-color:{PRIMARY_BG_COLOR};padding:20px;border-radius:10px;margin-bottom:25px;box-shadow: 0 4px 12px 0 rgba(0,0,0,0.3);">
        <h1 style="color:{PRIMARY_TEXT_COLOR};text-align:center;font-family: 'Verdana', sans-serif;">🚲 Aplikasi Prediksi Permintaan Sewa Sepeda</h1>
        <h4 style="color:{ACCENT_COLOR};text-align:center;font-family: 'Verdana', sans-serif;">Dipersembahkan oleh Tim COGNIDATA</h4>
    </div>
    """

HTML_FOOTER = f"""
    <div style="padding:10px;margin-top:40px;text-align:center;">
        <p style="color:grey;font-size:0.9em;">
            &copy; {datetime.date.today().year} Tim COGNIDATA - Prediksi Permintaan Sepeda
        </p>
    </div>
    """
# ===================================================================================
# Fungsi Utama Aplikasi
# ===================================================================================
def main():
    stc.html(HTML_BANNER, height=170)
    
    menu_options = {
        "🏠 Beranda": show_homepage,
        "⚙️ Aplikasi Prediksi": run_prediction_app,
        "📖 Info Model": show_model_info_page
    }
    
    st.sidebar.title("Navigasi Aplikasi")
    choice = st.sidebar.radio("", list(menu_options.keys()), label_visibility="collapsed")

    # Jalankan fungsi halaman yang dipilih
    if pipeline_model is not None or choice == "🏠 Beranda" or choice == "📖 Info Model":
        menu_options[choice]()
    elif pipeline_model is None and choice == "⚙️ Aplikasi Prediksi":
        st.error("Model prediksi tidak dapat dimuat. Halaman prediksi tidak dapat ditampilkan.")
        st.markdown("Silakan periksa file model dan coba lagi, atau hubungi administrator.")
    
    stc.html(HTML_FOOTER, height=70)

# ===================================================================================
# Halaman Beranda
# ===================================================================================
def show_homepage():
    st.markdown("## Selamat Datang di Dasbor Prediksi Permintaan Sepeda!")
    st.markdown("""
    Aplikasi ini adalah alat bantu cerdas untuk memprediksi jumlah total sepeda yang kemungkinan akan disewa dalam satu jam tertentu. 
    Dengan memanfaatkan data historis dan model machine learning canggih, kami bertujuan untuk memberikan estimasi yang dapat diandalkan 
    untuk membantu Anda dalam perencanaan dan operasional bisnis berbagi sepeda.

    ---
    #### Mengapa Prediksi Ini Penting?
    - Optimalisasi Stok Sepeda
    - Efisiensi Operasional dan Penjadwalan Perawatan
    - Peningkatan Kepuasan Pelanggan dengan Ketersediaan Sepeda
    - Dasar Strategi Pemasaran dan Promosi

    ---
    #### Cara Kerja Aplikasi:
    1.  Pilih "**⚙️ Aplikasi Prediksi**" dari menu navigasi di sebelah kiri.
    2.  Masukkan detail parameter waktu, kondisi cuaca, dan lingkungan pada formulir yang disediakan.
    3.  Klik tombol "**Prediksi Sekarang**" untuk melihat estimasi jumlah sewa.
    
    Jelajahi juga halaman "**📖 Info Model**" untuk memahami lebih dalam tentang teknologi di balik prediksi ini.

    ---
    #### Sumber Data:
    Dataset yang digunakan dalam pengembangan model ini berasal dari kompetisi Kaggle:
    [Bike Sharing Demand - Kaggle](https://www.kaggle.com/competitions/bike-sharing-demand/data)
    """)
    
    st.image("https://img.freepik.com/free-photo/row-parked-rental-bikes_53876-63261.jpg", 
             caption="Inovasi Transportasi Perkotaan dengan Berbagi Sepeda", use_column_width=True)

# ===================================================================================
# Halaman Aplikasi Prediksi
# ===================================================================================
def run_prediction_app():
    st.markdown("## ⚙️ Masukkan Parameter untuk Prediksi")
    
    if pipeline_model is None:
        # Pesan error sudah ditangani di fungsi main atau saat load_pickled_model
        # Cukup return agar tidak melanjutkan eksekusi halaman ini
        return

    # --- Bagian Input Tanggal dan Waktu ---
    st.markdown("#### 📅 Informasi Waktu")
    col_date, col_time = st.columns([1, 1]) 
    with col_date:
        # Default ke hari berikutnya agar lebih relevan untuk prediksi masa depan
        input_date = st.date_input("Tanggal Prediksi", datetime.date.today() + datetime.timedelta(days=1), 
                                   min_value=datetime.date.today(), # Min_value agar tidak bisa memilih tanggal lalu
                                   help="Pilih tanggal untuk prediksi.")
    with col_time:
        input_time = st.time_input("Waktu Prediksi", datetime.time(10, 0), 
                                   help="Pilih waktu (jam & menit) untuk prediksi.", step=3600) # Step 1 jam
    dt_object = datetime.datetime.combine(input_date, input_time)
    
    is_working_day_auto = 1 if dt_object.weekday() < 5 else 0 
    workingday_display_text = "Hari Kerja" if is_working_day_auto == 1 else "Akhir Pekan/Libur"
    st.info(f"Prediksi untuk: **{dt_object.strftime('%A, %d %B %Y, pukul %H:%M')}** ({workingday_display_text})")
    
    st.markdown("---")

    st.markdown("#### 📋 Kondisi & Lingkungan")
    col_kondisi1, col_kondisi2, col_lingkungan = st.columns([2, 2, 2.5]) 

    with col_kondisi1: 
        st.markdown("##### Musim & Liburan")
        season_options = {1: "Musim Semi", 2: "Musim Panas", 3: "Musim Gugur", 4: "Musim Dingin"} # Penyesuaian Kaggle: 1=spring, 2=summer, 3=fall, 4=winter
        current_month = dt_object.month
        if current_month in [3, 4, 5]: default_season = 1
        elif current_month in [6, 7, 8]: default_season = 2
        elif current_month in [9, 10, 11]: default_season = 3
        else: default_season = 4 # 12, 1, 2
        
        season = st.selectbox("Musim", options=list(season_options.keys()), 
                              format_func=lambda x: f"{season_options[x]} (Kode: {x})", 
                              index=list(season_options.keys()).index(default_season),
                              key="season_select")
        
        holiday = st.radio("Hari Libur Nasional?", (0, 1), format_func=lambda x: "Ya" if x == 1 else "Tidak", 
                           index=0, horizontal=True, key="holiday_radio")

    with col_kondisi2: 
        st.markdown("##### Status Hari & Cuaca")
        workingday = st.radio("Hari Kerja Aktual?", (0, 1), 
                              format_func=lambda x: "Ya" if x == 1 else "Tidak", 
                              index=is_working_day_auto, horizontal=True, key="workingday_radio",
                              help=f"Terdeteksi otomatis sebagai '{workingday_display_text}', Anda bisa mengubahnya jika perlu (misal: hari cuti bersama).")
        
        weather_options = {1: "Cerah/Sedikit Berawan", 2: "Kabut/Berawan Sebagian", 3: "Hujan/Salju Ringan", 4: "Cuaca Ekstrem (Hujan Lebat/Badai)"}
        weather = st.selectbox("Kondisi Cuaca", options=list(weather_options.keys()), 
                               format_func=lambda x: f"{weather_options[x]} (Kode: {x})", 
                               index=0, key="weather_select")

    with col_lingkungan: 
        st.markdown("##### Parameter Lingkungan")
        temp = st.number_input("Suhu (°C)", min_value=-20.0, max_value=50.0, value=25.0, step=0.5, format="%.1f", key="temp_input")
        humidity = st.slider("Kelembapan (%)", min_value=0, max_value=100, value=60, step=1, key="humidity_slider")
        windspeed = st.number_input("Kecepatan Angin (km/jam)", min_value=0.0, max_value=80.0, value=10.0, step=0.1, format="%.1f", key="windspeed_input")

    st.markdown("---")
    
    if st.button("Prediksi Jumlah Sewa Sekarang!", use_container_width=True, type="primary", key="predict_button_main"):
        # DataFrame input untuk model
        # Kolom 'atemp' akan dihapus oleh pipeline jika memang ada dalam latihannya
        # Jika tidak, dan model Anda tidak membutuhkannya (berdasarkan pra-pemrosesan), tidak perlu disertakan
        input_data_dict = {
            'datetime': [dt_object], # Model pipeline akan mengekstrak fitur dari datetime ini
            'season': [season], 
            'holiday': [holiday],
            'workingday': [workingday], 
            'weather': [weather], 
            'temp': [temp],
            'humidity': [humidity], 
            'windspeed': [windspeed]
            # 'atemp' tidak disertakan karena biasanya sangat berkorelasi dengan 'temp' dan mungkin sudah di-drop saat pra-pemrosesan
        }
        
        # Buat DataFrame hanya dengan kolom yang benar-benar ada saat pelatihan pipeline
        # Jika Anda tahu persis kolom input mentah yang dibutuhkan pipeline (sebelum pra-pemrosesan di dalamnya),
        # gunakan itu.
        input_features_df = pd.DataFrame(input_data_dict)
            
        st.markdown("#### Hasil Prediksi")
        try:
            # Prediksi menggunakan pipeline (yang akan menangani pra-pemrosesan)
            prediction_log = pipeline_model.predict(input_features_df)
            
            # Transformasi balik dari log ke skala asli
            predicted_count_original = np.expm1(prediction_log[0])
            
            # Pastikan tidak ada nilai negatif dan bulatkan
            predicted_count_final = max(0, int(round(predicted_count_original)))
            
            st.metric(label="Estimasi Jumlah Sewa Sepeda", value=f"{predicted_count_final} unit")

            if predicted_count_final < 50:
                st.info("Saran: Permintaan diprediksi rendah. Pertimbangkan promosi atau alokasi sepeda minimal.")
            elif predicted_count_final < 250:
                st.success("Saran: Permintaan diprediksi sedang. Pastikan ketersediaan sepeda cukup.")
            else:
                st.warning("Saran: Permintaan diprediksi tinggi. Siapkan ketersediaan sepeda ekstra dan pertimbangkan penempatan strategis.")
        except Exception as e:
            st.error(f"Gagal membuat prediksi: {e}")
            st.error("Pastikan input sesuai dan model telah dilatih dengan fitur yang benar.")
            
#====================================================================================#
# Halaman Informasi Model
#====================================================================================#
def show_model_info_page():
    st.markdown("## 📖 Informasi Detail Model Prediksi")
    st.markdown(f"""
    Model prediktif yang menjadi tulang punggung aplikasi ini adalah **XGBoost Regressor** yang dipaketkan dalam pipeline Scikit-learn.
    Pipeline ini dikembangkan dengan inspirasi dari alur kerja PyCaret, namun untuk deployment, pipeline finalnya disimpan dan digunakan secara mandiri dengan Scikit-learn untuk dependensi yang lebih ramping.

    #### Arsitektur & Pra-pemrosesan (dalam Pipeline):
    Model yang Anda gunakan (`{MODEL_FILENAME}`) adalah **keseluruhan pipeline pra-pemrosesan Scikit-learn dan model XGBoost** yang telah di-*fit* pada data historis. Proses yang ditangani oleh pipeline ini kemungkinan mencakup:
    - **Ekstraksi Fitur Waktu**: Dari kolom `datetime` (misalnya jam, hari, bulan, tahun, hari dalam seminggu, hari dalam setahun).
    - **Rekayasa Fitur Siklikal**: Transformasi sin/cos untuk fitur waktu periodik (jam, bulan, hari dalam seminggu) untuk menangkap sifat siklusnya.
    - **Penanganan Pencilan (Winsorizing)**: Untuk fitur seperti `humidity` dan `windspeed` jika diterapkan.
    - **Scaling Fitur Numerik**: Menggunakan `StandardScaler` atau metode serupa.
    - **Encoding Fitur Kategorikal**: Menggunakan `OneHotEncoder` untuk fitur seperti `season`, `weather`, `holiday`, `workingday`, dan fitur waktu kategorikal (`hour_val`, `month_val`, `weekday_val`, `year_cat`).
    - **Transformasi Target**: Variabel target (`count`) di-log-transformasi (`log1p`) sebelum pelatihan untuk menormalkan distribusinya. Prediksi dari model juga dalam skala log dan kemudian di-inverse-transform (`expm1`) kembali ke skala jumlah sewa asli di aplikasi ini.

    #### Sumber Data Acuan:
    Model ini dikembangkan berdasarkan konsep dan data dari kompetisi Kaggle:
    [Bike Sharing Demand - Kaggle](https://www.kaggle.com/competitions/bike-sharing-demand/data)

    #### Performa Model (Contoh dari Sesi Pelatihan Awal):
    *Metrik di bawah ini adalah contoh dari sesi pelatihan dan bisa bervariasi tergantung pada set validasi yang digunakan.*
    - **MAPE (Mean Absolute Percentage Error) pada Skala Asli**: Sekitar **21.82%**
    - **RMSLE (Root Mean Squared Logarithmic Error) pada Skala Asli**: Sekitar **0.2691**
    - **R² (R-squared) pada Skala Asli**: Sekitar **0.9612**
    
    *Performa pada data baru dapat bervariasi.*
    """)
    
    if pipeline_model is not None:
        st.markdown("#### Detail Pipeline dan Parameter Estimator Inti (XGBoost):")
        st.write("Struktur Pipeline Model:")
        st.write(pipeline_model) # Menampilkan struktur pipeline

        try:
            # Mengakses model XGBoost sebenarnya dari dalam pipeline scikit-learn
            actual_model_estimator = None
            if hasattr(pipeline_model, 'steps'): # Jika ini objek Pipeline
                final_step_estimator = pipeline_model.steps[-1][1] # Asumsi langkah terakhir adalah model
                if hasattr(final_step_estimator, 'regressor') and hasattr(final_step_estimator.regressor, 'get_params'): # Jika dibungkus (misal, oleh TransformedTargetRegressor jika tidak di-handle manual)
                     actual_model_estimator = final_step_estimator.regressor
                elif hasattr(final_step_estimator, 'get_params'): # Jika langkah terakhir adalah model itu sendiri
                    actual_model_estimator = final_step_estimator
            
            if actual_model_estimator and hasattr(actual_model_estimator, 'get_params'):
                st.markdown("Parameter Model XGBoost (dari pipeline):")
                st.json(actual_model_estimator.get_params(), expanded=False)
            else:
                st.warning("Tidak dapat mengekstrak parameter model XGBoost secara detail dari pipeline.")
        except Exception as e:
            st.warning(f"Terjadi kesalahan saat mencoba menampilkan parameter model: {e}")
    else:
        st.warning("Objek pipeline model tidak tersedia.")
    
    st.info("Untuk detail teknis lebih lanjut mengenai proses pelatihan dan validasi, silakan merujuk pada dokumentasi pengembangan internal Tim COGNIDATA.")

#====================================================================================#
# Menjalankan Aplikasi
#====================================================================================#
if __name__ == "__main__":
    if pipeline_model is None:
        st.error("KRITIS: GAGAL MEMUAT MODEL PREDIKSI SAAT APLIKASI DIMULAI.")
        st.markdown(f"Pastikan file model `{MODEL_FILENAME}` ada di direktori yang sama dengan `app.py` dan dapat diakses.")
        # Tidak menggunakan st.stop() di sini agar footer masih bisa tampil jika diinginkan,
        # tapi fungsi main akan menangani tampilan halaman jika model None.
    main()
