import numpy as np                                                 #for numerical values 
import pandas as pd                                                #for data handling

fake_df=pd.read_csv("Fake.csv")                                    #load dataset
true_df=pd.read_csv("True.csv")
fake_df.head()                                                     #EDA
print(fake_df.info())
print(fake_df.describe())
true_df.head()
print(true_df.info())
print(true_df.describe())
V=true_df['title'].value_counts()                                 #Count how many times each title appears
print(V)
S=true_df['title'].value_counts(normalize=True) * 100             # % of how many times each title appears
print(S)
T=true_df.isnull().sum()                                          #Count of missing values in each column
print(T)
U=(true_df.isnull().sum()/len(true_df))*100                       #Percentage of missing values in each column
print(U)
true_df.info()
fake_df["label"]=0                                                #False=0
true_df["label"]=1                                                #True=1
data=pd.concat([fake_df,true_df])                                 # Combine  both true and false data 
data=data.sample(frac=1).reset_index(drop=True)                   #Shuffle data & reset index
print(data.head())
import nltk                                                       #NLP tools
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))                     # create stopword and lemmatizer
lemmatizer = WordNetLemmatizer()

def clean_text(text):                                            # Define cleaning function 
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)                       # remove all the words that are not a-z or A-Z
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

data["cleaned"] = data["text"].apply(clean_text)                 #Apply Cleaning
from sklearn.feature_extraction.text import TfidfVectorizer      #Convert text into numbers 

vectorizer = TfidfVectorizer(max_features=5000)                  # top 5000 words 

X = vectorizer.fit_transform(data["cleaned"])
y = data["label"]
from sklearn.model_selection import train_test_split            #Split Data

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
from sklearn.naive_bayes import MultinomialNB                  #NAIVE BAYES MODEL

nb = MultinomialNB()
nb.fit(X_train, y_train)

y_pred_nb = nb.predict(X_test)

from sklearn.svm import LinearSVC                              #SVM MODEL

svm = LinearSVC()
svm.fit(X_train, y_train)

y_pred_svm = svm.predict(X_test)
from sklearn.metrics import accuracy_score, classification_report

print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))      # Evaluation of models 
print(classification_report(y_test, y_pred_nb))

print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))
