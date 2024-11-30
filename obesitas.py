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

#open file csv
df = pd.read_csv('data_obesitas.csv')

def show_grafik():
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Age", "Gender", "Height", "Weight", "BMI", "PhysicalActivityLevel", "ObesityCategory"])
    with tab1:
        st.write("Grafik Umur")
        chart_age = pd.DataFrame(df, columns=["Age"])
        st.pyplot(chart_age)
    with tab2:
        st.write("Grafik Jenis Kelamin")
        chart_gender = pd.DataFrame(df, columns=["Gender"])
        st.pyplot(chart_gender)
    with tab3:
        st.write("Grafik Tinggi Badan")
        chart_height = pd.DataFrame(df, columns=["Height"])
        st.pyplot(chart_height)
    with tab4:
        st.write("Grafik Berat Badan")
        chart_weight = pd.DataFrame(df, columns=["Weight"])
        st.pyplot(chart_weight)
    with tab5:
        st.write("Grafik BMI")
        chart_bmi = pd.DataFrame(df, columns=["BMI"])
        st.pyplot(chart_bmi)
    with tab6:
        st.write("Grafik PhysicalActivityLevel")
        chart_level = pd.DataFrame(df, columns=["PhysicalActivityLevel"])
        st.pyplot(chart_level)
    with tab7:
        st.write("Grafik ObesityCategory")
        chart_category = pd.DataFrame(df, columns=["ObesityCategory"])
        st.pyplot(chart_category)


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

# Sidepyplot for input
st.sidepyplot.header("Input Data")
age = st.sidepyplot.number_input("Umur", min_value=1, max_value=100, value=30)
gender = st.sidepyplot.selectbox("Jenis Kelamin", ["Laki-Laki", "Perempuan"])
height = st.sidepyplot.number_input("Tinggi Badan (cm)", min_value=100.0, max_value=250.0, value=170.0)
weight = st.sidepyplot.number_input("Berat Badan (kg)", min_value=20.0, max_value=200.0, value=70.0)
physical_activity = st.sidepyplot.slider("Level Aktivitas Fisik (1: Rendah, 5: Tinggi)", 1, 5, 3)

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
label_encoder.fit(["Laki-Laki", "Perempuan"])  # Fit the label encoder with the same labels used during training
input_data["Gender"] = label_encoder.transform(input_data["Gender"])

# Define the label encoder for the categories (Normal weight, Obese, Overweight, Underweight)
category_encoder = LabelEncoder()
category_encoder.fit(["Underweight", "Normal weight", "Overweight", "Obese"])

# Display input data
st.subheader("Data Input")
st.write(input_data)

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

st.subheader("Dataset")
st.dataframe(df)

st.subheader("Grafik")
show_grafik()


