import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Obesity Prediction App", layout="wide")
st.title("\U0001F4CA Aplikasi Prediksi Tingkat Obesitas")

st.markdown("""
Masukkan data berikut untuk memprediksi tingkat obesitas Anda.
Model yang digunakan telah dilatih dan dioptimalkan menggunakan GridSearchCV.
""")

# Load model dan scaler
model = joblib.load("best_model_rf.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        Age = st.slider("Usia", 10, 100, 25)
        Height = st.number_input("Tinggi Badan (m)", value=1.70)
        Weight = st.number_input("Berat Badan (kg)", value=70)
        FCVC = st.slider("Frekuensi makan sayur (1-3)", 1.0, 3.0, 2.0)
    with col2:
        CH2O = st.slider("Konsumsi air harian (1-3)", 1.0, 3.0, 2.0)
        FAF = st.slider("Aktivitas fisik mingguan (0-3)", 0.0, 3.0, 1.0)
        TUE = st.slider("Waktu layar harian (0-2)", 0.0, 2.0, 1.0)
        NCP = st.slider("Jumlah makan utama/hari (1-4)", 1, 4, 3)

    submitted = st.form_submit_button("Prediksi")

if submitted:
    X_new = pd.DataFrame([[Age, Height, Weight, FCVC, CH2O, FAF, TUE, NCP]],
                         columns=["Age", "Height", "Weight", "FCVC", "CH2O", "FAF", "TUE", "NCP"])

    X_scaled = scaler.transform(X_new)
    y_pred = model.predict(X_scaled)
    prediction_label = label_encoder.inverse_transform(y_pred)[0]

    st.success(f"\U0001F4A1 Prediksi Tingkat Obesitas Anda: {prediction_label}")

    bmi = Weight / (Height ** 2)
    st.info(f"BMI Anda: {bmi:.2f}")

    if bmi < 18.5:
        kategori = "Underweight"
    elif 18.5 <= bmi < 25:
        kategori = "Normal"
    elif 25 <= bmi < 30:
        kategori = "Overweight"
    else:
        kategori = "Obese"

    st.markdown(f"**Kategori BMI berdasarkan WHO: {kategori}**")
