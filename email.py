import streamlit as st
import joblib

# Load saved model and feature transformers
model = joblib.load('model_ceas08_enhanced.pkl')
tfidf_vectorizer = joblib.load('vectorizer_ceas08.pkl')
additional_features_extractor = joblib.load('additional_features_extractor.pkl')

# Define labels
label_map = {0: 'Legitimate', 1: 'Spam', 2: 'Phishing'}

st.title("📧 CEAS_08 Email Spam & Phishing Detector")

email_text = st.text_area("Paste your email content here:")

if st.button("Predict"):
    if not email_text.strip():
        st.warning("Please enter email content to predict.")
    else:
        # Preprocess input text
        email_text_processed = email_text.lower().strip()
        
        # Extract TF-IDF features and additional handcrafted features
        X_tfidf = tfidf_vectorizer.transform([email_text_processed])
        X_additional = additional_features_extractor.transform([email_text_processed])
        
        # Combine features
        from scipy.sparse import hstack
        X_combined = hstack([X_tfidf, X_additional])
        
        # Predict
        prediction = model.predict(X_combined)[0]
        
        st.success(f"Prediction: {label_map[prediction]}")
