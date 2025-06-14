import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("ObesityDataSet (2).csv")

# Pilih fitur numerik utama dan target
selected_features = ["Age", "Height", "Weight", "FCVC", "CH2O", "FAF", "TUE", "NCP"]
df = df.dropna(subset=selected_features + ["NObeyesdad"])

# Konversi fitur ke numerik
for col in selected_features:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Hapus baris yang gagal dikonversi
df = df.dropna(subset=selected_features)

# Feature & target
X = df[selected_features]
y = df["NObeyesdad"]

# Encode label target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Simpan LabelEncoder
joblib.dump(le, "label_encoder.pkl")

# Standarisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Simpan Scaler
joblib.dump(scaler, "scaler.pkl")

# Split dan latih model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Simpan model
joblib.dump(model, "best_model_rf.pkl")

# Evaluasi
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
