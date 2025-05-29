import streamlit as st
import streamlit.components.v1 as stc
import pandas as pd
import numpy as np
import pickle
import datetime
import os
from scipy.stats.mstats import winsorize # Tambahkan ini jika belum ada

# ===================================================================================
# SALIN FUNGSI-FUNGSI PRA-PEMROSESAN DARI NOTEBOOK COLAB KE SINI
# ===================================================================================
def winsorize_series_robust(df_or_series, column_name=None, limits=(0.01, 0.01)):
    # ... (definisi lengkap fungsi seperti di notebook) ...
    if isinstance(df_or_series, pd.DataFrame):
        if column_name is None or column_name not in df_or_series.columns:
            raise ValueError("Jika input adalah DataFrame, column_name harus valid.")
        series_to_winsorize = df_or_series[column_name].copy()
    else: # Asumsi Series
        series_to_winsorize = df_or_series.copy()
    
    winsorized_array = winsorize(series_to_winsorize, limits=limits)
    
    if isinstance(df_or_series, pd.DataFrame):
        df_out = df_or_series.copy()
        df_out[column_name] = winsorized_array
        return df_out
    else:
        return pd.Series(winsorized_array, name=df_or_series.name, index=df_or_series.index)

def preprocess_initial_features(input_df):
    df = input_df.copy()
    if 'datetime' in df.columns:
        df['hour_val'] = df['datetime'].dt.hour
        df['month_val'] = df['datetime'].dt.month
        df['weekday_val'] = df['datetime'].dt.weekday
        df['day'] = df['datetime'].dt.day
        df['year_cat'] = df['datetime'].dt.year.astype(str)
        df['dayofyear'] = df['datetime'].dt.dayofyear
        # Jangan drop datetime di sini jika pipeline Anda masih memerlukannya untuk langkah lain
        # Jika ColumnTransformer Anda hanya bekerja pada fitur turunan, maka bisa di-drop setelahnya.
        # Namun, jika pipeline Anda menangani datetime, biarkan saja.
        # Untuk amannya, karena pipeline Anda dilatih dengan fitur turunan, kita akan drop datetime
        # SETELAH semua fitur turunan dibuat di app.py, sebelum ke pipeline.
    if 'atemp' in df.columns:
        df = df.drop('atemp', axis=1, errors='ignore')
    return df

def create_cyclical_features(input_df):
    df = input_df.copy()
    if 'hour_val' in df.columns:
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_val']/24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_val']/24.0)
    if 'month_val' in df.columns:
        df['month_sin'] = np.sin(2 * np.pi * df['month_val']/12.0)
        df['month_cos'] = np.cos(2 * np.pi * df['month_val']/12.0)
    if 'weekday_val' in df.columns:
        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday_val']/7.0)
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday_val']/7.0)
    return df

# FunctionTransformer untuk winsorizing (perlu didefinisikan juga di app.py jika pipeline merujuknya)
# atau terapkan secara manual seperti di bawah
winsorizer_humidity_ft_app = lambda df: winsorize_series_robust(df, column_name='humidity', limits=(0.01, 0.01))
winsorizer_windspeed_ft_app = lambda df: winsorize_series_robust(df, column_name='windspeed', limits=(0.05, 0.05))


# ... (sisa kode st.set_page_config, load_pickled_model, HTML templates, main, show_homepage) ...

# ===================================================================================
# Halaman Aplikasi Prediksi (DENGAN PERBAIKAN REKAYASA FITUR)
# ===================================================================================
def run_prediction_app():
    st.markdown("## ⚙️ Masukkan Parameter untuk Prediksi")
    
    if pipeline_model is None:
        return

    # ... (Bagian Input Tanggal dan Waktu tetap sama) ...
    st.markdown("#### 📅 Informasi Waktu")
    col_date, col_time = st.columns([1, 1]) 
    with col_date:
        input_date = st.date_input("Tanggal Prediksi", datetime.date.today() + datetime.timedelta(days=1), 
                                   min_value=datetime.date.today(),
                                   help="Pilih tanggal untuk prediksi.")
    with col_time:
        input_time = st.time_input("Waktu Prediksi", datetime.time(10, 0), 
                                   help="Pilih waktu (jam & menit) untuk prediksi.", step=3600)
    dt_object = datetime.datetime.combine(input_date, input_time)
    
    is_working_day_auto = 1 if dt_object.weekday() < 5 else 0 
    workingday_display_text = "Hari Kerja" if is_working_day_auto == 1 else "Akhir Pekan/Libur"
    st.info(f"Prediksi untuk: **{dt_object.strftime('%A, %d %B %Y, pukul %H:%M')}** ({workingday_display_text})")
    
    st.markdown("---")

    st.markdown("#### 📋 Kondisi & Lingkungan")
    col_kondisi1, col_kondisi2, col_lingkungan = st.columns([2, 2, 2.5]) 

    with col_kondisi1: 
        st.markdown("##### Musim & Liburan")
        season_options = {1: "Musim Semi", 2: "Musim Panas", 3: "Musim Gugur", 4: "Musim Dingin"}
        current_month = dt_object.month
        if current_month in [3, 4, 5]: default_season = 1
        elif current_month in [6, 7, 8]: default_season = 2
        elif current_month in [9, 10, 11]: default_season = 3
        else: default_season = 4
        
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
        if pipeline_model is not None:
            # 1. Buat DataFrame awal dari input pengguna
            input_data_dict = {
                'datetime': [dt_object], 
                'season': [season], 
                'holiday': [holiday],
                'workingday': [workingday], 
                'weather': [weather], 
                'temp': [temp],
                'humidity': [humidity], 
                'windspeed': [windspeed]
            }
            input_df_raw = pd.DataFrame(input_data_dict)

            # 2. Terapkan pra-pemrosesan dan rekayasa fitur seperti di notebook
            input_df_p1 = preprocess_initial_features(input_df_raw.copy())
            input_df_p2 = create_cyclical_features(input_df_p1)
            
            input_df_engineered = input_df_p2.copy() # Inisialisasi
            if 'humidity' in input_df_engineered.columns:
                input_df_engineered = winsorizer_humidity_ft_app(input_df_engineered)
            if 'windspeed' in input_df_engineered.columns:
                input_df_engineered = winsorizer_windspeed_ft_app(input_df_engineered)

            # 3. Pastikan kolom 'datetime' sudah tidak ada jika pipeline tidak mengharapkannya
            #    dan semua kolom yang dibutuhkan pipeline (setelah rekayasa fitur) sudah ada.
            #    Pipeline Anda dilatih pada X_train_engineered yang tidak memiliki 'datetime' asli lagi.
            #    Kolom seperti 'hour_val', 'month_val', dll, sudah dibuat.
            
            # Kolom yang diharapkan oleh preprocessor_ct di pipeline Anda
            # Ambil ini dari definisi numeric_features_for_scaling dan categorical_features_for_ohe di notebook Anda
            expected_cols_for_pipeline = [
                'temp', 'humidity', 'windspeed', 'day', 'dayofyear',
                'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 
                'weekday_sin', 'weekday_cos', 'season', 'holiday', 
                'workingday', 'weather', 'hour_val', 'month_val', 
                'weekday_val', 'year_cat'
            ]
            # Pastikan semua kolom ini ada di input_df_engineered
            # dan input_df_engineered hanya berisi kolom-kolom ini dalam urutan yang benar jika pipeline sensitif urutan
            # Untuk ColumnTransformer, urutan tidak terlalu penting selama nama kolomnya benar.
            
            # Buat DataFrame final yang akan dimasukkan ke pipeline
            # dengan memastikan semua kolom yang diharapkan ada
            # dan tidak ada kolom ekstra (seperti 'datetime' asli)
            input_features_for_pipeline = input_df_engineered.reindex(columns=expected_cols_for_pipeline, fill_value=0) # fill_value=0 mungkin perlu disesuaikan
            # atau pastikan X_train_engineered di notebook Anda tidak punya kolom 'datetime' saat fit
            
            st.markdown("#### Hasil Prediksi")
            try:
                prediction_log = pipeline_model.predict(input_features_for_pipeline) # Gunakan DataFrame yang sudah direkayasa
                predicted_count_original = np.expm1(prediction_log[0])
                predicted_count_final = max(0, int(round(predicted_count_original)))
                
                st.metric(label="Estimasi Jumlah Sewa Sepeda", value=f"{predicted_count_final} unit")

                if predicted_count_final < 50:
                    st.info("Saran: Permintaan diprediksi rendah.")
                elif predicted_count_final < 250:
                    st.success("Saran: Permintaan diprediksi sedang.")
                else:
                    st.warning("Saran: Permintaan diprediksi tinggi.")
            except Exception as e:
                st.error(f"Gagal membuat prediksi: {e}")
                st.error("Pastikan input sesuai dan model telah dilatih dengan fitur yang benar.")
                st.write("DataFrame yang dimasukkan ke model setelah rekayasa fitur:")
                st.dataframe(input_features_for_pipeline) # Tampilkan untuk debug
                st.write("Kolom yang diharapkan:", expected_cols_for_pipeline)

# ... (sisa kode show_model_info_page dan if __name__ == "__main__":) ...
