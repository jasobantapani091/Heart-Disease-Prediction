import streamlit as st
import pandas as pd
import pickle
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Model ---
@st.cache_resource
def load_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model.pkl")
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please run the training script first.")
        return None

model = load_model()

# --- Custom Styling ---
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #FF0000;
        box-shadow: 0px 4px 10px rgba(255, 75, 75, 0.4);
        transform: scale(1.02);
    }
    .prediction-box-good {
        padding: 30px;
        border-radius: 15px;
        background: linear-gradient(135deg, #1f4037, #99f2c8);
        color: white;
        text-align: center;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.2);
    }
    .prediction-box-bad {
        padding: 30px;
        border-radius: 15px;
        background: linear-gradient(135deg, #cb2d3e, #ef473a);
        color: white;
        text-align: center;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# --- Title and Description ---
st.title("🫀 Heart Disease Risk Predictor")
st.markdown("""
Welcome to the AI-powered Heart Disease Risk Predictor. 
Please enter your health metrics in the sidebar to receive an instant assessment based on a machine learning model.
""")
st.divider()

# --- Sidebar Inputs ---
st.sidebar.header("📊 Enter Health Data")
st.sidebar.markdown("Adjust the sliders and dropdowns below:")

def user_input_features():
    age = st.sidebar.slider("Age", 20, 100, 50)
    sex = st.sidebar.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    cp = st.sidebar.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3], 
                              format_func=lambda x: [
                                  "0: Typical Angina", 
                                  "1: Atypical Angina", 
                                  "2: Non-anginal Pain", 
                                  "3: Asymptomatic"
                              ][x])
    trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
    chol = st.sidebar.slider("Serum Cholestoral (mg/dl)", 100, 600, 200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "True" if x == 1 else "False")
    restecg = st.sidebar.selectbox("Resting Electrocardiographic Results", options=[0, 1, 2],
                                   format_func=lambda x: [
                                       "0: Normal",
                                       "1: ST-T Wave Abnormality",
                                       "2: Left Ventricular Hypertrophy"
                                   ][x])
    thalach = st.sidebar.slider("Maximum Heart Rate Achieved", 60, 220, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    oldpeak = st.sidebar.slider("ST Depression (oldpeak)", 0.0, 6.2, 1.0, step=0.1)
    slope = st.sidebar.selectbox("Slope of Peak Exercise ST Segment", options=[0, 1, 2],
                                 format_func=lambda x: ["0: Upsloping", "1: Flat", "2: Downsloping"][x])
    ca = st.sidebar.slider("Number of Major Vessels (0-4) Colored by Flourosopy", 0, 4, 0)
    thal = st.sidebar.selectbox("Thalassemia (thal)", options=[0, 1, 2, 3],
                                format_func=lambda x: [
                                    "0: Error/Null",
                                    "1: Fixed Defect",
                                    "2: Normal",
                                    "3: Reversable Defect"
                                ][x])
    
    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# --- Main Display ---
st.subheader("📝 Your Input Summary")
st.dataframe(input_df.style.set_properties(**{'background-color': '#1E1E1E', 'color': 'white'}), hide_index=True)

st.write("")

if st.button("🔮 Predict Risk"):
    if model is not None:
        with st.spinner("Analyzing data..."):
            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)
            
            st.divider()
            
            if prediction[0] == 1:
                st.markdown(f"""
                <div class="prediction-box-bad">
                    <h2 style="color: white;">⚠️ High Risk Profile Detected</h2>
                    <p style="font-size: 18px;">The model predicts a <b>{prediction_proba[0][1]*100:.1f}%</b> probability of heart disease presence.</p>
                    <p><i>We strongly advise consulting with a healthcare professional for a comprehensive medical evaluation.</i></p>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f"""
                <div class="prediction-box-good">
                    <h2 style="color: white;">✅ Low Risk Profile Detected</h2>
                    <p style="font-size: 18px;">The model predicts a <b>{prediction_proba[0][0]*100:.1f}%</b> probability of a healthy heart.</p>
                    <p><i>Keep up the good work and maintain a healthy lifestyle!</i></p>
                </div>
                """, unsafe_allow_html=True)
                st.snow()
                
        # Feature Importance section
        st.write("")
        st.subheader("📊 What influences this prediction?")
        importances = model.feature_importances_
        feature_names = input_df.columns
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=True)
        
        st.bar_chart(importance_df.set_index('Feature'), color="#FF4B4B")
        
    else:
        st.warning("Prediction unavailable due to missing model.")

# --- Footer ---
st.markdown("""
<div style="text-align: center; margin-top: 50px; color: gray; font-size: 12px;">
    <i>Disclaimer: This application is for demonstration purposes only and does not provide medical advice. 
    Always consult a qualified healthcare provider.</i>
</div>
""", unsafe_allow_html=True)
