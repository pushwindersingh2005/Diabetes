import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺", layout="centered")

st.title("Diabetes Prediction")
st.markdown("Enter the patient's information and press **Predict** to check the likelihood of diabetes.")

# Load model and scaler
@st.cache_resource
def load_assets():
    try:
        with open("diabetes_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"File not found: {e}. Make sure both diabetes_model.pkl and scaler.pkl exist.")
        return None, None
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None

model, scaler = load_assets()

# Use a form to gather inputs
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
        glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=300, value=120)
        bp = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
        skin = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
    with col2:
        insulin = st.number_input("Insulin (mu U/ml)", min_value=0, max_value=900, value=80)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.2f")
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=5.0, value=0.5, format="%.3f")
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=30, step=1)

    submit = st.form_submit_button("Predict")

if submit:
    if model is None or scaler is None:
        st.warning("Model or scaler not loaded. See error message above.")
    else:
        try:
            # Prepare input array
            X = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
            
            # Standardize the input using the same scaler from training
            X_scaled = scaler.transform(X)
            
            # Make prediction
            pred = model.predict(X_scaled)
            
            # Get probability if available
            prob = None
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X_scaled)[0][1]

            # Display results
            if pred[0] == 1:
                if prob is not None:
                    st.error(f"⚠️ Prediction: **Diabetic** — probability = {prob*100:.2f}%")
                else:
                    st.error("⚠️ Prediction: **Diabetic**")
            else:
                if prob is not None:
                    st.success(f"✅ Prediction: **Non-diabetic** — probability of diabetes = {prob*100:.2f}%")
                else:
                    st.success("✅ Prediction: **Non-diabetic**")

            # Show details
            with st.expander("Show prediction details"):
                st.write(f"**Raw prediction class:** {pred[0]}")
                if prob is not None:
                    st.write(f"**Probability of diabetes:** {prob:.4f}")
                st.write(f"**Standardized input:** {X_scaled[0]}")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.exception(e)

st.markdown("---")
st.caption("Model expects features in this order: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age")