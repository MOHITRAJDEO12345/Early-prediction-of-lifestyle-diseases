import numpy as np
import joblib
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Load models (Ensure file extensions and paths are correct)
diabetes_model = pickle.load(open('diabetes/diabetes_model.sav', 'rb'))
parkinson_model = pickle.load(open('parkinson disease/parkinson_model.sav', 'rb'))
heart_model = pickle.load(open('heart disease/heart_model.sav', 'rb'))

# Load scaler for Parkinson’s model
scaler = pickle.load(open('parkinson disease/scalar.sav', 'rb'))

# Sidebar navigation
with st.sidebar:
    selected = option_menu(
        'Multiple Disease Prediction System',
        ['Home', 'Diabetes Prediction', 'Heart Disease Prediction', 'Parkinson Disease Prediction'],
        icons=['house', 'activity', 'heart', 'person'],
        default_index=0
    )

# Utility function to safely convert input to float
def safe_float(value, default=0.0):
    try:
        return float(value)
    except ValueError:
        return default  # Assigns default value if conversion fails


# 🚀 Home Page
if selected == 'Home':
    st.title("🩺 Multiple Disease Prediction System")
    st.markdown("""
    ## Welcome to the **AI-Powered Health Prediction System**!  
    This tool helps in predicting the likelihood of three major diseases using **Machine Learning**:
    
    - **🩸 Diabetes**
    - **❤️ Heart Disease**
    - **🧠 Parkinson's Disease**
    
    👉 Select a prediction model from the sidebar to proceed!  
    ⚠ **Disclaimer:** This tool is for educational purposes and should not replace medical advice.
    """)

    # st.image("https://www.cdc.gov/diabetes/images/library/spotlights/diabetes-tests.jpg", width=700)

    
# 🩸 Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML (SVC)')
    st.markdown("""
    This model predicts the likelihood of **Diabetes** based on various health parameters.  
    Please enter the required medical details below and click **"Diabetes Test Result"** to get the prediction.
    """)

    # Create columns for better input organization
    col1, col2 = st.columns(2)

    # Column 1 Inputs
    with col1:
        Pregnancies = safe_float(st.text_input("Number of Pregnancies", "0"))
        Glucose = safe_float(st.text_input("Glucose Level", "100"))
        BloodPressure = safe_float(st.text_input("Blood Pressure", "80"))
        SkinThickness = safe_float(st.text_input("Skin Thickness", "20"))

    # Column 2 Inputs
    with col2:
        Insulin = safe_float(st.text_input("Insulin Level", "79"))
        BMI = safe_float(st.text_input("BMI (Body Mass Index)", "25.0"))
        DiabetesPedigreeFunction = safe_float(st.text_input("Diabetes Pedigree Function", "0.5"))
        Age = st.number_input("Enter Age", min_value=10, max_value=100, value=30, step=1)

    # Button for Prediction
    if st.button('Diabetes Test Result'):
        try:
            input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, 
                                    Insulin, BMI, DiabetesPedigreeFunction, Age]])
            diab_prediction = diabetes_model.predict(input_data)

            result = "The person is diabetic" if diab_prediction[0] == 1 else "The person is not diabetic"
            st.success(result)

        except Exception as e:
            st.error(f"❌ Error: {e}")

            
# ❤️ Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML (Logistic Regression)')
    st.markdown("""
    This model predicts the likelihood of **Heart Disease** based on clinical parameters.  
    Please enter the required medical details below and click **"Heart Disease Test Result"** to get the prediction.
    """)

    # Create columns for input organization
    col1, col2, col3 = st.columns(3)

    with col1:
        age = safe_float(st.text_input("Age", "45"))
        sex = safe_float(st.text_input("Sex (1 = Male, 0 = Female)", "1"))
        cp = safe_float(st.text_input("Chest Pain Type (0-3)", "2"))
        trestbps = safe_float(st.text_input("Resting Blood Pressure (mm Hg)", "130"))

    with col2:
        chol = safe_float(st.text_input("Serum Cholesterol (mg/dL)", "200"))
        fbs = safe_float(st.text_input("Fasting Blood Sugar > 120 mg/dL (1 = Yes, 0 = No)", "0"))
        restecg = safe_float(st.text_input("Resting ECG Results (0-2)", "1"))
        thalach = safe_float(st.text_input("Maximum Heart Rate Achieved", "150"))

    with col3:
        exang = safe_float(st.text_input("Exercise-Induced Angina (1 = Yes, 0 = No)", "0"))
        oldpeak = safe_float(st.text_input("ST Depression Induced by Exercise", "1.0"))
        slope = safe_float(st.text_input("Slope of Peak Exercise ST Segment (0-2)", "1"))
        ca = safe_float(st.text_input("Number of Major Vessels (0-4) Colored by Fluoroscopy", "0"))
        thal = safe_float(st.text_input("Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)", "2"))

    if st.button('Heart Disease Test Result'):
        try:
            input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, 
                                    exang, oldpeak, slope, ca, thal]])
            heart_prediction = heart_model.predict(input_data)

            result = "The person has heart disease" if heart_prediction[0] == 1 else "The person does not have heart disease"
            st.success(result)

        except Exception as e:
            st.error(f"❌ Error: {e}")


# 🧠 Parkinson’s Disease Prediction Page
if selected == 'Parkinson Disease Prediction':
    st.title('Parkinson Disease Prediction using ML (SVC)')
    st.markdown("""
    This model uses **voice frequency & amplitude data** to predict **Parkinson’s disease**.
    """)

    # Inputs for Parkinson's prediction
    inputs = [safe_float(st.text_input(f"Feature {i+1}", "0.0")) for i in range(22)]

    if st.button('Parkinson Disease Test Result'):
        try:
            input_data = np.array([inputs])
            std_data = scaler.transform(input_data)
            parkinson_prediction = parkinson_model.predict(std_data)

            result = "The person has Parkinson’s disease" if parkinson_prediction[0] == 1 else "The person does not have Parkinson’s disease"
            st.success(result)

        except Exception as e:
            st.error(f"❌ Error: {e}")
