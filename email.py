import streamlit as st
import joblib

# Load CEAS_08-trained model & vectorizer
model = joblib.load('model_ceas08.pkl')
vectorizer = joblib.load('vectorizer_ceas08.pkl')

label_map = {0: "Legitimate", 1: "Spam", 2: "Phishing"}

st.title("📧 CEAS_08 Email Spam & Phishing Detector")

email_text = st.text_area("Paste your email content here:")

if st.button("Predict"):
    if not email_text.strip():
        st.warning("Please enter email content to predict.")
    else:
        # Preprocess & vectorize input
        X_input = vectorizer.transform([email_text.lower().strip()])

        # Predict with the loaded model
        pred = model.predict(X_input)[0]

        # Display result
        st.success(f"Prediction: {label_map[pred]}")
