# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# -----------------------------------------------------
# Load Model
# -----------------------------------------------------
# Ganti path ini sesuai file model kamu
MODEL_PATH = 'best_rf_model.pkl'

with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

# -----------------------------------------------------
# Layout Streamlit
# -----------------------------------------------------
st.title("Obesity Level Prediction App")
st.markdown("""
Aplikasi ini memprediksi tingkat obesitas berdasarkan data yang Anda inputkan.
""")

# -----------------------------------------------------
# Input User
# -----------------------------------------------------
# Numerical Inputs
st.header("Masukkan Data Anda:")
age = st.number_input('Usia', min_value=1, max_value=120, value=25)
height = st.number_input('Tinggi Badan (m)', min_value=1.0, max_value=2.5, value=1.70)
weight = st.number_input('Berat Badan (kg)', min_value=20, max_value=300, value=70)
fcvc = st.slider('Frekuensi makan sayuran (1-3)', 1, 3, 2)
ncp = st.slider('Jumlah makan besar/hari', 1, 4, 3)
ch2o = st.slider('Konsumsi air (1-3)', 1, 3, 2)
faf = st.slider('Aktivitas fisik (0-3)', 0, 3, 1)
tue = st.slider('Waktu layar (0-2)', 0, 2, 1)

# Categorical Inputs
gender = st.selectbox('Jenis Kelamin', ['Male', 'Female'])
family_history = st.selectbox('Riwayat Keluarga Kelebihan Berat Badan?', ['yes', 'no'])
favc = st.selectbox('Sering makan makanan tinggi kalori?', ['yes', 'no'])
caec = st.selectbox('Makan camilan di antara waktu makan?', ['no', 'Sometimes', 'Frequently', 'Always'])
smoke = st.selectbox('Merokok?', ['yes', 'no'])
scc = st.selectbox('Memantau asupan kalori?', ['yes', 'no'])
calc = st.selectbox('Konsumsi alkohol?', ['no', 'Sometimes', 'Frequently', 'Always'])
mtrans = st.selectbox('Transportasi utama?', [
    'Automobile', 'Motorbike', 'Bike', 'Public_Transportation', 'Walking'
])

# -----------------------------------------------------
# Buat DataFrame Input
# -----------------------------------------------------
input_data = pd.DataFrame({
    'Age': [age],
    'Height': [height],
    'Weight': [weight],
    'FCVC': [fcvc],
    'NCP': [ncp],
    'CH2O': [ch2o],
    'FAF': [faf],
    'TUE': [tue],
    'Gender': [gender],
    'family_history_with_overweight': [family_history],
    'FAVC': [favc],
    'CAEC': [caec],
    'SMOKE': [smoke],
    'SCC': [scc],
    'CALC': [calc],
    'MTRANS': [mtrans]
})

# -----------------------------------------------------
# Prediksi
# -----------------------------------------------------
if st.button('Prediksi'):
    prediction = model.predict(input_data)
    st.subheader('Hasil Prediksi:')
    st.success(f"Tingkat Obesitas Anda: **{prediction[0]}**")
