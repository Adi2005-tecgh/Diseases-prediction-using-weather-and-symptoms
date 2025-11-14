  # 🧑‍⚕️ Weather-Based Disease Prediction

This project predicts the most probable disease using weather conditions, symptoms, and personal details. The machine-learning model analyzes temperature, humidity, wind speed, age, gender, and selected symptoms to identify likely diseases and support early awareness and preventive action.

---

## 📌 Features

- Predicts diseases using:
  - Weather factors  
  - User-reported symptoms  
  - Basic demographics  
- Machine learning classification model  
- Clean and interactive Streamlit UI  
- Shows Top-5 probable diseases  
- Includes saved model, feature names, and label encoder  
- Dataset visualizations and evaluation metrics included  

---

## 🧰 Tech Stack

- Python  
- Streamlit  
- Scikit-learn  
- Joblib  
- Pandas / NumPy  
- Matplotlib / Seaborn  

---

## 📁 Project Structure
project/
│── app.py
│── models/
│ ├── weather_disease_model.joblib
│ ├── feature_names.joblib
│ └── label_encoder.joblib
│── outputs/
│ ├── figures/
│ └── tables/
│── requirements.txt
│── README.md


---

## 🚀 How to Run the Project

### 1️⃣ Create & Activate Virtual Environment

**Windows**
python -m venv .venv
.venv\Scripts\activate


**Mac / Linux**
python3 -m venv .venv
source .venv/bin/activate


### 2️⃣ Install Dependencies
pip install -r requirements.txt

### 3️⃣ Run the Streamlit App
streamlit run app.py


Open in browser:
👉 http://localhost:8501

---

## 📦 Model Training Summary

The model was trained using a dataset containing:

- Symptoms  
- Weather conditions  
- Age & gender  
- Disease prognosis labels  

Generated during training:

- Trained model pipeline (`weather_disease_model.joblib`)  
- Encoded feature list (`feature_names.joblib`)  
- Label encoder (`label_encoder.joblib`)  
- Model performance tables & confusion matrices  

---

## 👥 Group Members

- **Shravan Ghodke** – PRN: *202301070168*  
- **Aditya Sonakanalli** – PRN: *202301070175*  
- **Samiksha Hubale** – PRN: *202301070178*  
- **Aditi Nalawade** – PRN: *202301070179*  

---

## ⚠️ Disclaimer

This is not a medical diagnostic tool.  
It provides probabilistic predictions and should not replace professional medical advice.



