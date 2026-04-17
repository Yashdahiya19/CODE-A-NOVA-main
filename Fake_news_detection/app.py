import streamlit as st
import pickle
import re
import os
import nltk

# Download NLTK data first
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = pickle.load(open(os.path.join(BASE_DIR, "fake_news_model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb"))

# NLP setup
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Same cleaning function used in training
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

st.title("Political Fake News Detector")
with st.expander("About this model"):
    st.write("""
    This fake news detection model was trained using a dataset containing
    mainly US political news articles. The model uses NLP preprocessing,
    TF-IDF vectorization, and a Support Vector Machine classifier.

    Because of dataset bias, the model performs best on US political news
    and may not generalize well to other countries or domains.
    """)

text = st.text_area("Enter News Article")

if st.button("Predict"):
    cleaned_text = clean_text(text)
    text_vector = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_vector)

    if prediction[0] == 1:
        st.success("Prediction: Real News")
    else:
        st.error("Prediction: Fake News")