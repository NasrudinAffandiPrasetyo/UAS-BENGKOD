import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Obesity Level Prediction App", layout="wide")

st.title("\U0001F4A1 Obesity Level Prediction App")
st.markdown("""
Aplikasi ini memprediksi tingkat obesitas berdasarkan data yang Anda inputkan.
Silakan isi formulir di bawah ini, lalu klik tombol prediksi.
""")

@st.cache_data
def load_model():
    df = pd.read_csv("ObesityDataSet (1).csv")
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col])
    X = df.drop("NObeyesdad", axis=1)
    y = df["NObeyesdad"]
    model = RandomForestClassifier()
    model.fit(X, y)
    return model, le, X.columns

model, le, feature_cols = load_model()

st.markdown("## \U0001F4DD Masukkan Data Anda:")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        Age = st.slider("Usia", 10, 100, 25)
        Height = st.number_input("Tinggi Badan (m)", value=1.70)
        Weight = st.number_input("Berat Badan (kg)", value=70)
        FCVC = st.slider("Frekuensi makan sayuran (1-3)", 1.0, 3.0, 2.0)
        NCP = st.slider("Jumlah makan besar/hari", 1, 4, 3)
        CH2O = st.slider("Konsumsi air (1-3)", 1.0, 3.0, 2.0)
        FAF = st.slider("Aktivitas fisik (0-3)", 0.0, 3.0, 1.0)
    with col2:
        TUE = st.slider("Waktu layar (0-2 jam)", 0.0, 2.0, 1.0)
        Gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
        family_history = st.selectbox("Riwayat Keluarga Kelebihan Berat Badan?", ["yes", "no"])
        FAVC = st.selectbox("Sering makan makanan tinggi kalori?", ["yes", "no"])
        CAEC = st.selectbox("Makan camilan di antara waktu makan?", ["no", "Sometimes", "Frequently", "Always"])
        SMOKE = st.selectbox("Merokok?", ["yes", "no"])
        SCC = st.selectbox("Memantau asupan kalori?", ["yes", "no"])
        CALC = st.selectbox("Konsumsi alkohol?", ["no", "Sometimes", "Frequently", "Always"])
        MTRANS = st.selectbox("Transportasi utama?", ["Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"])

    submitted = st.form_submit_button("\U0001F680 Prediksi Obesitas")

if submitted:
    input_df = pd.DataFrame([{
        "Age": Age,
        "Height": Height,
        "Weight": Weight,
        "FCVC": FCVC,
        "NCP": NCP,
        "CH2O": CH2O,
        "FAF": FAF,
        "TUE": TUE,
        "Gender": Gender,
        "family_history_with_overweight": family_history,
        "FAVC": FAVC,
        "CAEC": CAEC,
        "SMOKE": SMOKE,
        "SCC": SCC,
        "CALC": CALC,
        "MTRANS": MTRANS
    }])

    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_cols]

    for col in input_df.select_dtypes(include="object").columns:
        input_df[col] = le.fit_transform(input_df[col])

    prediction = model.predict(input_df)[0]
    st.success(f"\U0001F4CA Hasil Prediksi Obesitas Anda: {prediction}")
