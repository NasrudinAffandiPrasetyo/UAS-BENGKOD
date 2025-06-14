import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Analisis Obesitas", layout="wide")

st.title("UAS Bengkel Koding: Analisis Obesitas")
st.markdown("""
Dataset ini berisi data mengenai kebiasaan makan, aktivitas fisik, dan karakteristik individu
untuk memprediksi tingkat obesitas seseorang.
""")

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("ObesityDataSet (1).csv")
    return df

df = load_data()

# ----------------- Tampilan Dataset -----------------
st.subheader("📄 Dataset")
st.dataframe(df)

# ----------------- Visualisasi Distribusi -----------------
st.subheader("📊 Distribusi Obesitas (NObeyesdad)")
st.bar_chart(df["NObeyesdad"].value_counts())

# ----------------- Korelasi -----------------
st.subheader("📈 Korelasi Antara Variabel Numerik")
num_cols = df.select_dtypes(include=["float64", "int64"]).columns

if len(num_cols) >= 2:
    corr = df[num_cols].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
else:
    st.warning("Tidak cukup data numerik untuk menampilkan heatmap korelasi.")

# ----------------- Preprocessing -----------------
st.subheader("🛠️ Prediksi Obesitas")
df_model = df.copy()
le = LabelEncoder()
for col in df_model.select_dtypes(include='object').columns:
    df_model[col] = le.fit_transform(df_model[col])

X = df_model.drop("NObeyesdad", axis=1)
y = df_model["NObeyesdad"]
model = RandomForestClassifier()
model.fit(X, y)

# ----------------- Form Input -----------------
st.markdown("### Masukkan Data Untuk Prediksi")

col1, col2, col3 = st.columns(3)

with col1:
    Gender = st.selectbox("Gender", df["Gender"].unique())
    Age = st.slider("Age", 10, 100, 25)
    Height = st.number_input("Height (m)", value=1.70)
    Weight = st.number_input("Weight (kg)", value=70)

with col2:
    family_history = st.selectbox("Family History", df["family_history_with_overweight"].unique())
    FAVC = st.selectbox("Frequent High Caloric Food", df["FAVC"].unique())
    FCVC = st.slider("Vegetable Consumption (0-3)", 0.0, 3.0, 2.0)
    NCP = st.slider("Meal Frequency", 1, 4, 3)

with col3:
    CAEC = st.selectbox("Food Consumption Between Meals", df["CAEC"].unique())
    TUE = st.slider("Time Using Technology (hrs)", 0, 4, 2)
    MTRANS = st.selectbox("Transport", df["MTRANS"].unique())
    CALC = st.selectbox("Alcohol Consumption", df["CALC"].unique())

input_df = pd.DataFrame({
    "Gender": [Gender],
    "Age": [Age],
    "Height": [Height],
    "Weight": [Weight],
    "family_history_with_overweight": [family_history],
    "FAVC": [FAVC],
    "FCVC": [FCVC],
    "NCP": [NCP],
    "CAEC": [CAEC],
    "TUE": [TUE],
    "MTRANS": [MTRANS],
    "CALC": [CALC]
})

# Tambah kolom yang kosong
for col in X.columns:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[X.columns]

# Encoding
for col in input_df.select_dtypes(include='object').columns:
    input_df[col] = le.fit_transform(input_df[col])

# Prediksi
if st.button("Prediksi Obesitas"):
    prediction = model.predict(input_df)[0]
    st.success(f"Hasil Prediksi: {prediction}")

st.markdown("---")
st.info("Dibuat oleh Nasrudin Affandi Prasetyo untuk tugas UAS Bengkel Koding ✨")
