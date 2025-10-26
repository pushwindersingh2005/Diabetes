# 🩺 Diabetes Prediction App

This is a simple Streamlit web application that predicts the likelihood of a person having diabetes based on medical parameters such as glucose level, BMI, insulin, age, and more.  
This project uses a pre-trained **Machine Learning model** serialized using `pickle`.

---

## 🚀 Features

- Interactive web interface using **Streamlit**
- Real-time diabetes prediction
- Displays probability of being diabetic (if supported by the model)
- User-friendly and mobile-responsive layout
- Error handling for missing model/scaler files

---

## 🧠 Model Overview

The app uses a machine learning model trained on the **PIMA Indians Diabetes Dataset** (commonly used for diabetes classification).  
It requires the following input features:

1. Pregnancies  
2. Glucose  
3. Blood Pressure  
4. Skin Thickness  
5. Insulin  
6. BMI  
7. Diabetes Pedigree Function  
8. Age  

---

## 🛠️ Technologies Used

- **Python 3.x**
- **Streamlit**
- **NumPy**
- **Scikit-learn**
- **Pickle** (for model serialization)

---

## 📦 Project Structure

📁 diabetes-prediction-app
│
├── app.py # Main Streamlit application
├── diabetes_model.pkl # Trained ML model
├── scaler.pkl # StandardScaler used during training
├── requirements.txt # List of required Python packages
└── README.md # Project documentation

---

## 🧾 Output

- **Non-Diabetic: ✅**
- **Diabetic: ⚠️**

 Displays prediction result and probability (if supported by the model).

---

## 💡 Future Improvements

- **Add model retraining capability**

- **Include data visualization for feature impact**

- **Deploy to Streamlit Cloud or Hugging Face Spaces**

---

## 🤝 Contributing

Pull requests are welcome!

For major changes, please open an issue first to discuss what you would like to change.

---
