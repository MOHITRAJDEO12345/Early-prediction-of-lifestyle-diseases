import os
import numpy as np
import pandas as pd
import joblib
import pickle
import streamlit as st
import seaborn as sns
from streamlit_option_menu import option_menu
import time
import matplotlib.pyplot as plt
import json
import google.generativeai as genai
from dotenv import load_dotenv



# Set page config with icon
st.set_page_config(page_title="Disease Prediction", page_icon="🩺", layout="wide")

# Load models (Ensure file extensions and paths are correct)
diabetes_model = pickle.load(open('diabetes/diabetes_model.sav', 'rb'))
parkinson_model = pickle.load(open('parkinson disease/parkinson_model.sav', 'rb'))
heart_model = pickle.load(open('heart disease/heart_model.sav', 'rb'))

# Load scaler for Parkinson’s model
scaler = pickle.load(open('parkinson disease/scalar.sav', 'rb'))



# Sidebar navigation with icons and colors
with st.sidebar:
    st.title("🩺 Disease Prediction")
    
    selected = option_menu(
        menu_title="Navigation",
        options=['Home', 'Diabetes Prediction', 'Heart Disease Prediction', 'Parkinson Disease Prediction', 'Data Visualization', 'Chat with us'],
        icons=['house', 'activity', 'heart', 'person'],
        menu_icon="cast",
        default_index=0,
       styles={
            "container": {"padding": "5px", "background-color": "#111111"},  # Darker background
            "icon": {"color": "#FF0000", "font-size": "20px"},  # White icons
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "color": "#FFFFFF"},  # White text
            "nav-link-selected": {"background-color": "#FF0000", "color": "#FFFFFF"},
       },
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
    """)

    with st.expander("🚀 Quick Start Guide"):
        st.write("""
        1. Select a disease prediction model from the sidebar.
        2. Enter your medical details in the input fields.
        3. Click the prediction button to get your result.
        """)

    st.markdown("""
        ### Learn More
        - [What is Diabetes?](https://www.example.com)
        - [Understanding Heart Disease](https://www.example.com)
        - [Parkinson’s Disease Overview](https://www.example.com)
        """)


    rating = st.slider("⭐ Rate this app", 1, 5)
    if st.button('Submit Rating'):
        st.success(f"Thank you for rating us {rating} stars!")


    feedback = st.text_area("💬 Provide Feedback", placeholder="How was your experience?")
    if st.button('Submit Feedback'):
        st.success("Thank you for your feedback!")

    st.markdown("""
        ### Contact Us
        Have questions? Email us at [support@diseaseprediction.com](mailto:support@diseaseprediction.com).
        """)    
    st.markdown("---")
    st.markdown("""
    **Disclaimer:** This app is for educational purposes only and should not replace professional medical advice.
    """)

# 🩸 Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('🩸 Diabetes Prediction using ML (SVC)')
    st.image("https://cdn-icons-png.flaticon.com/512/2919/2919950.png", width=100)

    st.markdown("""
    This model predicts the likelihood of **Diabetes** based on various health parameters.  
    Please enter the required medical details below and click **"Diabetes Test Result"** to get the prediction.
    """)



    # Create columns for better input organization
    col1, col2 = st.columns(2)

    with col1:
        Pregnancies = safe_float(st.text_input("Number of Pregnancies", "0"))
        Glucose = safe_float(st.text_input("Glucose Level", "100"))
        BloodPressure = safe_float(st.text_input("Blood Pressure", "80"))
        SkinThickness = safe_float(st.text_input("Skin Thickness", "20"))

    with col2:
        Insulin = safe_float(st.text_input("Insulin Level", "79"))
        BMI = safe_float(st.text_input("BMI (Body Mass Index)", "25.0"))
        DiabetesPedigreeFunction = safe_float(st.text_input("Diabetes Pedigree Function", "0.5"))
        Age = st.number_input("Enter Age", min_value=10, max_value=100, value=30, step=1)

    with col1:
        if st.button('Diabetes Test Result'):
            try:
                input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
                
                with st.spinner("⏳ Predicting... Please wait..."):
                    time.sleep(2)  # Simulating delay (remove in actual use)
                    diab_prediction = diabetes_model.predict(input_data)

                
                
                result = "🛑 The person is diabetic" if diab_prediction[0] == 1 else "✅ The person is not diabetic"
                if diab_prediction[0] == 0:
                    st.balloons()  # Or use st.confetti() if you install the library
                st.success(result)

            except Exception as e:
                st.error(f"❌ Error: {e}")


    with col2:
        if st.button('View Diabetes Graphs'):
            st.markdown("### Diabetes Graphs")
            # Placeholder for diabetes graphs
            fig, ax = plt.subplots()
            ax.plot([0, 1, 2, 3, 4], [0, 1, 4, 9, 16], label="Glucose Level")
            ax.set_title("Diabetes Graph Example")
            ax.set_xlabel("Time")
            ax.set_ylabel("Glucose Level")
            st.pyplot(fig)

# ❤️ Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.title('❤️ Heart Disease Prediction using ML (Logistic Regression)')
    st.image("https://cdn-icons-png.flaticon.com/512/3332/3332679.png", width=100)

    st.markdown("""
    This model predicts the likelihood of **Heart Disease** based on clinical parameters.
    """)

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
        ca = safe_float(st.text_input("Number of Major Vessels (0-4)", "0"))
        thal = safe_float(st.text_input("Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)", "2"))
    with col1:
        if st.button('Heart Disease Test Result'):
            try:
                input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
                with st.spinner("⏳ Predicting... Please wait..."):
                    time.sleep(2)  # Simulating delay (remove in actual use)
                heart_prediction = heart_model.predict(input_data)

                result = "🛑 The person has heart disease" if heart_prediction[0] == 1 else "✅ The person does not have heart disease"
                if heart_prediction[0] == 0:
                    st.balloons()  # Or use st.confetti() if you install the library
                st.success(result)

            except Exception as e:
                st.error(f"❌ Error: {e}")

    with col3:
        if st.button('View Heart Disease Graphs'):
            st.markdown("### Heart Disease Graphs")
            # Placeholder for heart disease graphs
            fig, ax = plt.subplots()
            ax.plot([0, 1, 2, 3, 4], [0, 1, 2, 3, 4], label="Heart Rate")
            ax.set_title("Heart Disease Graph Example")
            ax.set_xlabel("Time")
            ax.set_ylabel("Heart Rate")
            st.pyplot(fig)

# 🧠 Parkinson’s Disease Prediction Page
if selected == 'Parkinson Disease Prediction':
    st.title('🧠 Parkinson Disease Prediction using ML (SVC)')
    st.image("https://cdn-icons-png.flaticon.com/512/4221/4221843.png", width=100)
    st.markdown("This model uses **voice frequency & amplitude data** to predict **Parkinson’s disease**.")

    # Create two columns for input fields
    col1, col2 = st.columns(2)
    
    # Column 1 Inputs
    with col1:
        MDVP_Fo = safe_float(st.text_input("MDVP:Fo (Hz) - Fundamental Frequency", "197.076"))
        MDVP_Fhi = safe_float(st.text_input("MDVP:Fhi (Hz) - Highest Frequency", "206.896"))
        MDVP_Flo = safe_float(st.text_input("MDVP:Flo (Hz) - Lowest Frequency", "192.055"))
        MDVP_Jitter = safe_float(st.text_input("MDVP:Jitter (%) - Frequency Variation", "0.00289"))
        MDVP_Jitter_Abs = safe_float(st.text_input("MDVP:Jitter (Abs)", "0.00001"))
        MDVP_RAP = safe_float(st.text_input("MDVP:RAP", "0.00166"))
        MDVP_PPQ = safe_float(st.text_input("MDVP:PPQ", "0.00168"))
        Jitter_DDP = safe_float(st.text_input("Jitter:DDP", "0.00498"))
        MDVP_Shimmer = safe_float(st.text_input("MDVP:Shimmer (%) - Amplitude Variation", "0.01098"))
    
    # Column 2 Inputs
    with col2:
        MDVP_Shimmer_dB = safe_float(st.text_input("MDVP:Shimmer (dB)", "0.09700"))
        Shimmer_APQ3 = safe_float(st.text_input("Shimmer:APQ3", "0.00563"))
        Shimmer_APQ5 = safe_float(st.text_input("Shimmer:APQ5", "0.00680"))
        MDVP_APQ = safe_float(st.text_input("MDVP:APQ", "0.00802"))
        Shimmer_DDA = safe_float(st.text_input("Shimmer:DDA", "0.01689"))
        NHR = safe_float(st.text_input("NHR - Noise to Harmonics Ratio", "0.00339"))
        HNR = safe_float(st.text_input("HNR - Harmonic to Noise Ratio", "26.775"))
        RPDE = safe_float(st.text_input("RPDE - Recurrence Period Density Entropy", "0.422229"))
        DFA = safe_float(st.text_input("DFA - Detrended Fluctuation Analysis", "0.741367"))

    # Additional row for remaining inputs
    col3, col4 = st.columns(2)
    
    with col3:
        spread1 = safe_float(st.text_input("Spread1", "-7.348300"))
        spread2 = safe_float(st.text_input("Spread2", "0.177551"))
    
    with col4:
        D2 = safe_float(st.text_input("D2 - Correlation Dimension", "1.743867"))
        PPE = safe_float(st.text_input("PPE - Pitch Period Entropy", "0.085569"))

    parkinson_diagnostic = ""

    with col3:
        if st.button('Parkinson Disease Test Result'):
            try:
                # Prepare input data for prediction
                input_data = np.array([[
                    MDVP_Fo, MDVP_Fhi, MDVP_Flo, MDVP_Jitter, MDVP_Jitter_Abs, MDVP_RAP, MDVP_PPQ,
                    Jitter_DDP, MDVP_Shimmer, MDVP_Shimmer_dB, Shimmer_APQ3, Shimmer_APQ5, MDVP_APQ,
                    Shimmer_DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE
                ]])

                # Standardize input data using the saved scaler
                std_data = scaler.transform(input_data)

                with st.spinner("⏳ Predicting... Please wait..."):
                    time.sleep(2)  # Simulating delay (remove in actual use)

                # Make prediction using the trained model
                parkinson_prediction = parkinson_model.predict(std_data)

                # Display result
                
                parkinson_diagnostic = "🟥 The person has Parkinson’s disease" if parkinson_prediction[0] == 1 else "✅ The person does not have Parkinson’s disease"
                if parkinson_prediction[0] == 0:
                    st.balloons()
                st.success(parkinson_diagnostic)

            except Exception as e:
                st.error(f"❌ Error: {e}")

    with col4:
        if st.button('View Parkinson’s Disease Graphs'):
            st.markdown("### Parkinson’s Disease Graphs")
            # Placeholder for Parkinson’s disease graphs
            fig, ax = plt.subplots()
            ax.plot([0, 1, 2, 3, 4], [0, 1, 0.5, 0.75, 0.25], label="Voice Frequency")
            ax.set_title("Parkinson’s Disease Graph Example")
            ax.set_xlabel("Time")
            ax.set_ylabel("Voice Frequency")
            st.pyplot(fig)

if selected == 'Data Visualization':
    # st.set_page_config(page_title="Data Visualizer",
    #                 page_icon="📊", layout="centered")
    st.title(" 📊 Data Visualization")

    working_dir = os.path.dirname(os.path.abspath(__file__))

    folder_path = f"{working_dir}/data_csv"

    files_list = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    selected_file = st.selectbox("Select a file", files_list, index=None)

    if selected_file:

        file_path = os.path.join(folder_path, selected_file)

        df = pd.read_csv(file_path)

        columns = df.columns.tolist()

        col1, col2 = st.columns(2)

        with col1:
            st.write("")
            st.write(df.head())

        with col2:
            x_axis = st.selectbox("Select X-axis", options=columns + ["None"])
            y_axis = st.selectbox("Select Y-axis", options=columns + ["None"])

            plot_list = ["Line Plot", "Bar Plot", "Scatter Plot", "Histogram", "Box Plot", "Distribution Plot", "Count Plot", "Pair Plot"]

            selected_plot = st.selectbox("Select a plot", options=plot_list, index=None)

            # st.write(x_axis)
            # st.write(y_axis)
            # st.write(selected_plot)

        if st.button("Generate Plot"):

            fig, ax = plt.subplots(figsize=(6,4))

            if selected_plot == "Line Plot":
                sns.lineplot(x=x_axis, y=y_axis, data=df, ax=ax)

            elif selected_plot == "Bar Plot":
                sns.barplot(x=x_axis, y=y_axis, data=df, ax=ax)
            
            elif selected_plot == "Scatter Plot":
                sns.scatterplot(x=x_axis, y=y_axis, data=df, ax=ax)
            
            elif selected_plot == "Histogram":
                sns.histplot(df[x_axis], ax=ax)
            
            elif selected_plot == "Box Plot":
                sns.boxplot(x=x_axis, y=y_axis, data=df, ax=ax)

            elif selected_plot == "Distribution Plot":
                sns.kdeplot(df[x_axis], ax=ax)
            
            elif selected_plot == "Count Plot":
                sns.countplot(x=x_axis, data=df, ax=ax)
            
            elif selected_plot == "Pair Plot":
                sns.pairplot(df, ax=ax)

            ax.tick_params(axis="x", labelsize=10)
            ax.tick_params(axis="y", labelsize=10)

            plt.title(f"{selected_plot} of {x_axis} vs {y_axis}", fontsize=12)
            plt.xlabel(x_axis, fontsize=10)
            plt.ylabel(y_axis, fontsize=10)

            st.pyplot(fig)


if selected == 'Chat with us':
    st.title("👩‍💻 Chat with us")
    st.markdown("### Let's chat about your health concerns!")
    st.write("Please enter your message below and we'll get back to you shortly.")

    load_dotenv()

# Get the Gemini API key
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        st.error("API Key not found! Please check your .env file.")
        st.stop()

# Configure Gemini API
    genai.configure(api_key=api_key)

    # Streamlit UI for Health Chatbot
    selected = "Chat with us"  # Ensure this matches your navigation logic

    if selected == "Chat with us":
        # Custom Styling
        st.markdown("""
            <style>
                body { background-color: #1e1e1e; color: #ffffff; }
                .stChatMessage { font-size: 16px; }
                .prompt-box { 
                    background-color: #333333; 
                    padding: 10px; 
                    border-radius: 8px; 
                    font-size: 14px; 
                    font-family: monospace;
                    margin-bottom: 10px;
                }
                .copy-button {
                    background-color: #444444;
                    color: white;
                    border: none;
                    padding: 5px 10px;
                    border-radius: 5px;
                    font-size: 12px;
                    cursor: pointer;
                    margin-top: 5px;
                }
                .copy-button:hover {
                    background-color: #666666;
                }
            </style>
        """, unsafe_allow_html=True)

        st.title("🩺 Health Specialist Chatbot")
        st.markdown("### Get expert advice on Heart Disease, Diabetes, and Parkinson’s!")
        st.write("Ask me about symptoms, diet, exercise, and lifestyle recommendations.")

        # Predefined Prompts
        st.markdown("#### 💡 Quick Prompts (Click to Copy)")
        
        prompt_options = {
            "🍏 Diet Advice": "What foods should I eat if I have diabetes?",
            "🏋️ Physical Well-being": "What exercises are good for heart health?",
            "🩺 Symptoms Check": "I have fatigue and dizziness. Could it be related to heart disease?"
        }

        for label, prompt in prompt_options.items():
            st.markdown(f"""
            <div class="prompt-box">
                <strong>{label}</strong><br>
                {prompt}
            </div>
            """, unsafe_allow_html=True)

            st.code(prompt, language="text")

        # Initialize chat history if not present
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Display previous chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # User input field
        user_prompt = st.chat_input("Ask about your health...")

        if user_prompt:
            # Display user message
            st.chat_message("user").markdown(user_prompt)
            st.session_state.chat_history.append({"role": "user", "content": user_prompt})

            # Gemini API request
            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(user_prompt)

            if response and hasattr(response, "text"):
                assistant_response = response.text
            else:
                assistant_response = "I'm sorry, I couldn't generate a response."

            st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

            # Display assistant's response
            with st.chat_message("assistant"):
                st.markdown(assistant_response)