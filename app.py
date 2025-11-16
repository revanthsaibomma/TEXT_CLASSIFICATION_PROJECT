# -*- coding: utf-8 -*-

import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')

# --------------------------
# Load Model & Vectorizer
# --------------------------
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# --------------------------
# Cleaning Function
# --------------------------
def clean_text(text):
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)
    text = text.lower()
    text = " ".join([w for w in text.split() if w not in stopwords.words("english")])
    return text

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Sentiment Classification", page_icon="💬")

st.title("💬 Sentiment Classification App")
st.write("Enter text and the model will predict whether it is Positive, Negative, or Neutral.")

user_input = st.text_area("✍️ Enter your text:")

if st.button("Classify Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        clean = clean_text(user_input)
        vec = vectorizer.transform([clean]).toarray()
        pred = model.predict(vec)[0]

        # Display Result
        if pred == "positive":
            st.success("Result: 😊 **Positive Sentiment**")
        elif pred == "negative":
            st.error("Result: 😠 **Negative Sentiment**")
        else:
            st.info("Result: 😐 **Neutral Sentiment**")

st.markdown("---")
st.caption("Developed by Himani Sharma | NLP Lab Mini Project (BCS-DS-658)")
