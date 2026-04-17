import numpy as np
import pandas as pd

# Load dataset
fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

# Add labels
fake_df["label"] = 0
true_df["label"] = 1

# Combine datasets
data = pd.concat([fake_df, true_df])
data = data.sample(frac=1).reset_index(drop=True)

# Handle missing values
data = data.fillna("")

# Combine title + text (important improvement)
data["content"] = data["title"] + " " + data["text"]

# NLP libraries
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# Apply cleaning
data["cleaned"] = data["content"].apply(clean_text)

# TF-IDF Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=20000)
X = vectorizer.fit_transform(data["cleaned"])

y = data["label"]

# Train test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Naive Bayes
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()
nb.fit(X_train, y_train)

# Train SVM
from sklearn.svm import LinearSVC

svm = LinearSVC()
svm.fit(X_train, y_train)

# Evaluation
from sklearn.metrics import accuracy_score, classification_report

y_pred_nb = nb.predict(X_test)
y_pred_svm = svm.predict(X_test)

print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))

print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

# Prediction function
def predict_news(news):
    news = clean_text(news)
    news_vector = vectorizer.transform([news])
    prediction = svm.predict(news_vector)

    if prediction[0] == 1:
        return "Real News"
    else:
        return "Fake News"

# Save model and vectorizer
import pickle

pickle.dump(svm, open("fake_news_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

# Test prediction
test_news = "U.S. President Joe Biden said on Tuesday that the United States will continue to support Ukraine as the conflict with Russia continues. Speaking at the White House, Biden emphasized that the U.S. government is working closely with its European allies to provide economic and military assistance to Ukraine.The President stated that the administration has approved additional funding packages aimed at strengthening Ukraine’s defense capabilities and supporting humanitarian relief efforts. Officials from the Department of Defense confirmed that the aid will include advanced defensive systems and training for Ukrainian forces.Biden also reiterated the importance of international cooperation, noting that NATO members remain united in their response to the conflict. He added that the United States will continue diplomatic efforts to reduce tensions while maintaining strong support for Ukraine’s sovereignty and territorial integrity."
print(predict_news(test_news))