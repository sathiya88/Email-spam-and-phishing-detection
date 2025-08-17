import streamlit as st
import joblib

# Load saved model and TF-IDF vectorizer only
model = joblib.load('model_spam_phishing.pkl')
tfidf_vectorizer = joblib.load('tfidf_spam_phishing.pkl')

label_map = {0: 'Legitimate', 1: 'Spam', 2: 'Phishing'}

st.title("📧 Email Spam & Phishing Detector")

email_text = st.text_area("Paste your email content here:")

if st.button("Predict"):
    if not email_text.strip():
        st.warning("Please enter email content to predict.")
    else:
        email_text_processed = email_text.lower().strip()
        
        # Use only TF-IDF features for prediction
        X_tfidf = tfidf_vectorizer.transform([email_text_processed])
        
        prediction = model.predict(X_tfidf)[0]
        st.success(f"Prediction: {label_map[prediction]}")
