import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Obesity Level Prediction App", layout="wide")

st.title("\U0001F4A1 Obesity Level Prediction App")
st.markdown("""
Aplikasi ini memprediksi tingkat obesitas berdasarkan data yang Anda inputkan
menggunakan fitur numerik utama agar hasil prediksi lebih akurat dan logis.
""")

@st.cache_data
def load_model():
    df = pd.read_csv("ObesityDataSet (1).csv")

    # Gunakan hanya fitur numerik utama
    selected_features = ["Age", "Height", "Weight", "FCVC", "CH2O", "FAF", "TUE"]
    X = df[selected_features]
    y = df["NObeyesdad"]

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    return model, selected_features

model, selected_features = load_model()

st.markdown("## \U0001F4DD Masukkan Data Anda:")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        Age = st.slider("Usia", 10, 100, 25)
        Height = st.number_input("Tinggi Badan (m)", value=1.70)
        Weight = st.number_input("Berat Badan (kg)", value=70)
    with col2:
        FCVC = st.slider("Frekuensi makan sayuran (1-3)", 1.0, 3.0, 2.0)
        CH2O = st.slider("Konsumsi air (1-3)", 1.0, 3.0, 2.0)
        FAF = st.slider("Aktivitas fisik (0-3)", 0.0, 3.0, 1.0)
        TUE = st.slider("Waktu layar (0-2 jam)", 0.0, 2.0, 1.0)

    submitted = st.form_submit_button("\U0001F680 Prediksi Obesitas")

if submitted:
    input_df = pd.DataFrame([{
        "Age": Age,
        "Height": Height,
        "Weight": Weight,
        "FCVC": FCVC,
        "CH2O": CH2O,
        "FAF": FAF,
        "TUE": TUE
    }])

    bmi = Weight / (Height ** 2)
    st.info(f"BMI Anda: {bmi:.2f}")

    prediction = model.predict(input_df)[0]
    st.success(f"\U0001F4CA Hasil Prediksi Obesitas Anda: {prediction}")

    # Evaluasi berdasarkan BMI standar WHO
    if bmi < 18.5:
        kategori_bmi = "Underweight"
    elif 18.5 <= bmi < 25:
        kategori_bmi = "Normal"
    elif 25 <= bmi < 30:
        kategori_bmi = "Overweight"
    else:
        kategori_bmi = "Obese"

    st.markdown(f"**Kategori BMI berdasarkan WHO: {kategori_bmi}**")

    if (kategori_bmi == "Normal" and "Obesity" in prediction) or (kategori_bmi == "Underweight" and "Obesity" in prediction):
        st.warning("⚠️ Hasil prediksi tampaknya tidak sesuai dengan kategori BMI Anda. Model sebelumnya mungkin bias, kini difokuskan pada numerik.")
