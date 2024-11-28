import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load trained model
model_path = 'prediksi_obesitas.sav'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Title
st.title("Prediksi Kategori Obesitas")

# Deskripsi di bawah judul
st.markdown("""
    Aplikasi ini digunakan untuk memprediksi kategori obesitas berdasarkan data pribadi pengguna seperti umur, jenis kelamin, tinggi badan, 
    berat badan, dan tingkat aktivitas fisik. Hasil prediksi akan memberikan kategori obesitas berdasarkan rumus BMI (Body Mass Index), 
    yang mengklasifikasikan individu ke dalam kategori: Underweight, Normal weight, Overweight, atau Obese.  
    <br>
    <strong>Sumber data:</strong> <a href="https://www.kaggle.com/datasets/mrsimple07/obesity-prediction/data" target="_blank">https://www.kaggle.com/datasets/mrsimple07/obesity-prediction/data</a>
    <br>
    Dibuat oleh Seno Satrio_223307026
""", unsafe_allow_html=True)

# Sidebar for input
st.sidebar.header("Input Data")
age = st.sidebar.number_input("Umur", min_value=1, max_value=100, value=30)
gender = st.sidebar.selectbox("Jenis Kelamin", ["Male", "Female"])
height = st.sidebar.number_input("Tinggi Badan (cm)", min_value=100.0, max_value=250.0, value=170.0)
weight = st.sidebar.number_input("Berat Badan (kg)", min_value=20.0, max_value=200.0, value=70.0)
physical_activity = st.sidebar.slider("Level Aktivitas Fisik (1: Rendah, 5: Tinggi)", 1, 5, 3)

# Calculate BMI
bmi = weight / ((height / 100) ** 2)

# Create input dataframe
input_data = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "Height": [height],
    "Weight": [weight],
    "BMI": [bmi],
    "PhysicalActivityLevel": [physical_activity]
})

# Preprocess input data (make sure to use the same transformation as during training)
# Using LabelEncoder for 'Gender'
label_encoder = LabelEncoder()
label_encoder.fit(["Male", "Female"])  # Fit the label encoder with the same labels used during training
input_data["Gender"] = label_encoder.transform(input_data["Gender"])

# Define the label encoder for the categories (Normal weight, Obese, Overweight, Underweight)
category_encoder = LabelEncoder()
category_encoder.fit(["Underweight", "Normal weight", "Overweight", "Obese"])

# Make prediction
if st.button("Prediksi"):
    try:
        # Predict the category (assuming the model predicts the integer label for categories)
        prediction = model.predict(input_data)
        
        # Convert prediction to the corresponding category label
        prediction_label = category_encoder.inverse_transform(prediction)[0]
        
        st.subheader("Hasil Prediksi")
        st.write(f"Kategori Obesitas: **{prediction_label}**")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {str(e)}")

# Display input data
st.subheader("Data Input")
st.write(input_data)
