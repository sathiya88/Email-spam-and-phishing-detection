import streamlit as st
import pickle
import os

# -----------------------------
# 1. Page Config
# -----------------------------
st.set_page_config(
    page_title="Email Spam & Phishing Detector",
    page_icon="📧",
    layout="wide"
)

# -----------------------------
# 2. Load Model and Vectorizer
# -----------------------------
model_path = "spam_model.pkl"
vectorizer_path = "tfidf_vectorizer.pkl"

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    st.error("Model files not found! Please ensure 'spam_model.pkl' and 'tfidf_vectorizer.pkl' are in the same folder.")
    st.stop()

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

# -----------------------------
# 3. Dashboard Header
# -----------------------------
st.title("📧 Email Spam & Phishing Detection Dashboard")
st.markdown("""
This tool classifies incoming emails as **Spam** or **Not Spam (Legitimate)** using a trained ML model.
""")

# -----------------------------
# 4. Sidebar Information
# -----------------------------
st.sidebar.header("About the App")
st.sidebar.info("""
- **Model:** TF-IDF + ML Classifier  
- **Input:** Email text content  
- **Output:** Spam / Not Spam with confidence  
""")
st.sidebar.markdown("**Author:** Your Name")
st.sidebar.markdown("**Version:** 1.0")

# -----------------------------
# 5. Input Section
# -----------------------------
st.subheader("🔍 Check an Email")
email_text = st.text_area("Paste your email content here:", height=200, placeholder="Enter email text...")

if st.button("Classify Email"):
    if not email_text.strip():
        st.warning("Please enter some text before classification.")
    else:
        # Transform the text and predict
        features = vectorizer.transform([email_text])
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0].max() * 100 if hasattr(model, "predict_proba") else None

        # Show result
        if prediction == 1:
            st.error(f"🚨 This email is **SPAM / PHISHING**.")
        else:
            st.success(f"✅ This email is **NOT SPAM**.")

        if probability is not None:
            st.info(f"Model confidence: **{probability:.2f}%**")

# -----------------------------
# 6. Footer
# -----------------------------
st.markdown("---")
st.caption("© 2025 Email Spam Detection | Built with Streamlit")

