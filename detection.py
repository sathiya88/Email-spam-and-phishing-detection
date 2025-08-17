import streamlit as st
import joblib

# Load saved model & TF-IDF vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

label_map = {0: 'Legitimate', 1: 'Spam', 2: 'Phishing'}

st.title("📧 Email Spam & Phishing Detector")

email_text = st.text_area("Paste your email content here:")

if st.button("Predict"):
    if not email_text.strip():
        st.warning("Please enter email content to predict.")
    else:
        X_input = vectorizer.transform([email_text.lower()])
        pred = model.predict(X_input)[0]
        st.success(f"Prediction: {label_map[pred]}")

