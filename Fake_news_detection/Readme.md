#  Political Fake News Detector (ML Project)

A Machine Learning web application that detects whether a political news article is **Real** or **Fake** using Natural Language Processing (NLP) and a trained classification model.

The project uses **TF-IDF vectorization** and a **Support Vector Machine (SVM)** model to analyze text patterns and classify news articles.

---

# Features

- Detects whether a political news article is **Real or Fake**
- NLP-based preprocessing (cleaning, stopword removal, lemmatization)
- TF-IDF text vectorization
- Machine Learning classification using SVM
- Interactive web interface built with **Streamlit**



# Important Note

This model was trained mainly on a dataset containing **US political news articles**.

Because of this dataset bias:
- The model performs best on **US political news**
- Predictions may be **less accurate for news from other countries (e.g., India)**
- It may also perform poorly on **non-political topics** such as science, sports, or entertainment.

This project is built **for educational and demonstration purposes**.


# Machine Learning Pipeline

The model follows this pipeline:

1. Data Collection  
   - Fake and real political news datasets

2. Data Preprocessing
   - Lowercasing text
   - Removing punctuation
   - Stopword removal
   - Lemmatization

3. Feature Extraction
   - TF-IDF Vectorization

4. Model Training
   - Support Vector Machine (SVM)
   - Naive Bayes (tested for comparison)

5. Model Deployment
   - Streamlit web application

---

#  Technologies Used

- Python
- Scikit-learn
- Pandas
- NumPy
- NLTK
- Streamlit
- Pickle

---

# Project Structure
fake-news-detection/
│
├── app.py # Streamlit web application
├── fake_news_model.pkl # Trained ML model
├── vectorizer.pkl # TF-IDF vectorizer
├── requirements.txt # Required Python libraries
└── README.md # Project documentation

---

# ▶ Running the Project Locally

### 1️ Clone the repository

git clone https://github.com/Anshusharma-code/code-a-nova/Fake_news_detection.git

### 2️ Install dependencies
pip install -r requirements.txt

### 3  Run the Streamlit app
streamlit run app.py

The application will open in your browser.

---

#  Web App Interface

Users can:

- Enter a news article
- Click **Predict**
- The model will classify the article as **Real News** or **Fake News**

---

#  Future Improvements

Possible improvements for the project:

- Train on **global news datasets**
- Use **BERT or Transformer-based models**
- Add **confidence scores for predictions**
- Improve UI with more interactive elements
- Build an **API for integration with other applications**

---

#  Author: Yash Dahiya

B.Tech Student | Interested in AI & Machine Learning


#  Acknowledgment

This project was created as a learning project to explore **NLP, machine learning, and model deployment using Streamlit**.