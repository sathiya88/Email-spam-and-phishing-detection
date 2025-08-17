import streamlit as st
import joblib
from scipy.sparse import hstack

# Load saved model and feature transformers
model = joblib.load('model_spam_phishing.pkl')
tfidf_vectorizer = joblib.load('tfidf_spam_phishing.pkl')
additional_features_extractor = joblib.load('additional_features_extractor_spam_phishing.pkl')

label_map = {0: 'Legitimate', 1: 'Spam', 2: 'Phishing'}

st.title("📧 Email Spam & Phishing Detector")

email_text = st.text_area("Paste your email content here:")

if st.button("Predict"):
    if not email_text.strip():
        st.warning("Please enter email content to predict.")
    else:
        email_text_processed = email_text.lower().strip()
        X_tfidf = tfidf_vectorizer.transform([email_text_processed])
        X_additional = additional_features_extractor.transform([email_text_processed])
        X_combined = hstack([X_tfidf, X_additional])
        prediction = model.predict(X_combined)[0]
        st.success(f"Prediction: {label_map[prediction]}")
