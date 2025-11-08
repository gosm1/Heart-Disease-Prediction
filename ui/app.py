import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# ------------------------------
# Load Model and Feature Columns
# ------------------------------


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "final_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "..", "models", "feature_columns.pkl")

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURES_PATH)

# ------------------------------
# App Config & Title
# ------------------------------
st.set_page_config(page_title="❤️ Heart Disease Prediction", layout="centered")
st.title("💓 Heart Disease Prediction App")
st.write("Provide your health details below to predict the likelihood of heart disease.")

# ------------------------------
# Sidebar for User Input
# ------------------------------
st.sidebar.header("User Input Features")

def user_input_features():
    # Numeric inputs
    age = st.sidebar.slider("Age", 20, 100, 50)
    trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.sidebar.slider("Serum Cholesterol (mg/dl)", 100, 600, 240)
    thalch = st.sidebar.slider("Maximum Heart Rate Achieved", 70, 210, 150)
    oldpeak = st.sidebar.slider("ST Depression", 0.0, 6.0, 1.0)
    ca = st.sidebar.slider("Number of Major Vessels (0–3)", 0, 3, 0)

    # Categorical inputs
    sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
    cp = st.sidebar.selectbox("Chest Pain Type", ["typical angina","atypical angina","non-anginal","asymptomatic"])
    exang = st.sidebar.selectbox("Exercise Induced Angina", ["Yes", "No"])
    slope = st.sidebar.selectbox("Slope", ["upsloping", "flat", "downsloping"])
    thal = st.sidebar.selectbox("Thal", ["normal","reversable defect","fixed defect"])

    # Create dictionary of raw features
    data = {
        'age': age,
        'trestbps': trestbps,
        'chol': chol,
        'thalch': thalch,
        'oldpeak': oldpeak,
        'ca': ca,
        'sex_Male': 1 if sex=="Male" else 0,
        'cp_typical angina':1 if cp=="typical angina" else 0,
        'cp_atypical angina':1 if cp=="atypical angina" else 0,
        'cp_non-anginal':1 if cp=="non-anginal" else 0,
        'cp_asymptomatic':1 if cp=="asymptomatic" else 0,
        'exang_True':1 if exang=="Yes" else 0,
        'slope_upsloping':1 if slope=="upsloping" else 0,
        'slope_flat':1 if slope=="flat" else 0,
        'slope_downsloping':1 if slope=="downsloping" else 0,
        'thal_normal':1 if thal=="normal" else 0,
        'thal_reversable defect':1 if thal=="reversable defect" else 0,
        'thal_fixed defect':1 if thal=="fixed defect" else 0
    }

    df = pd.DataFrame(data, index=[0])

    # Add any missing columns from training
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns to match training
    df = df[feature_columns]
    return df

input_df = user_input_features()

st.subheader("🔹 Input Data Summary")
st.write(input_df)

# ------------------------------
# Prediction Section
# ------------------------------
if st.button("🔍 Predict"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    st.subheader("🧠 Prediction Result:")
    if prediction > 0:
        st.error("⚠️ The model predicts a *risk of heart disease.*")
    else:
        st.success("✅ The model predicts *no significant risk of heart disease.*")

    st.write(f"Prediction Confidence: {round(max(proba)*100, 2)}%")

# ------------------------------
# Data Visualization Section (Optional)
# ------------------------------
st.markdown("---")
st.subheader("📊 Explore Heart Disease Trends")

uploaded_file = st.file_uploader("Upload a Heart Disease dataset (CSV) to visualize trends", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", data.head())

    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
