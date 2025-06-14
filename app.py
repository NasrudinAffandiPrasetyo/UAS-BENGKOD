import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Analisis Obesitas", layout="wide")

st.title("UAS Bengkel Koding: Analisis Obesitas")
st.markdown("Dataset: ObesityDataSet")

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("ObesityDataSet (1).csv")
    return df

df = load_data()

st.subheader("📄 Dataset")
st.dataframe(df)

st.subheader("📊 Distribusi Obesitas (NObeyesdad)")
st.bar_chart(df["NObeyesdad"].value_counts())

st.subheader("📈 Korelasi Antara Variabel Numerik")
num_cols = df.select_dtypes(include=["float64", "int64"]).columns
corr = df[num_cols].corr()

fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

st.markdown("---")
st.info("Dibuat oleh Nasrudin Affandi Prasetyo untuk tugas UAS Bengkel Koding")

