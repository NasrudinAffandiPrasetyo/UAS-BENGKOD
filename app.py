import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Obesity Level Prediction App", layout="wide")

st.title("💡 Obesity Level Prediction App")
st.markdown("""
Aplikasi ini memprediksi tingkat obesitas berdasarkan data yang Anda inputkan.
Silakan isi formulir di bawah ini, lalu klik tombol prediksi.
""")

# Load Dataset & Latih Model
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

# --- Form Input ---
st.markdown("## 📝 Masukkan Data Anda:")

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
        TUE = st.slider("Waktu layar (0-2 jam)", 0.0, 2.0, 1
