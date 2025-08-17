import streamlit as st
import joblib
import re
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import csr_matrix, hstack

# Define the custom class again in the Streamlit app
class AdditionalFeaturesExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        spam_keywords = ['congratulations', 'winner', 'prize', 'urgent', 'bank', 'claim', 'lottery']
        suspicious_domains = ['.ru', 'xn--', '.xyz', '.top', '.biz', '.info']
        features = []

        for text in X:
            text_lower = text.lower()
            spam_word_count = sum(text_lower.count(word) for word in spam_keywords)
            url_count = len(re.findall(r"http[s]?://", text_lower))
            suspicious_domain_flag = int(any(domain in text_lower for domain in suspicious_domains))
            features.append([spam_word_count, url_count, suspicious_domain_flag])

        return csr_matrix(features)

# Load saved model and transformers
model = joblib.load('model_ceas08_enhanced.pkl')
tfidf_vectorizer = joblib.load('vectorizer_ceas08.pkl')
additional_features_extractor = joblib.load('additional_features_extractor.pkl')

# Map labels to names
label_map = {0: 'Legitimate', 1: 'Spam', 2: 'Phishing'}

# Streamlit UI
st.title("📧 CEAS_08 Email Spam & Phishing Detector")

email_text = st.text_area("Paste your email content here:")

if st.button("Predict"):
    if not email_text.strip():
        st.warning("Please enter email content to predict.")
    else:
        # Preprocess input
        email_text_processed = email_text.lower().strip()

        # Extract features
        X_tfidf = tfidf_vectorizer.transform([email_text_processed])
        X_additional = additional_features_extractor.transform([email_text_processed])

        # Combine features
        X_combined = hstack([X_tfidf, X_additional])

        # Predict
        prediction = model.predict(X_combined)[0]

        st.success(f"Prediction: {label_map[prediction]}")
