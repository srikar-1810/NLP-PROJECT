import re
import joblib
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = text.split()
    stop_words = set(stopwords.words("english"))
    lem = WordNetLemmatizer()
    cleaned = [lem.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(cleaned)

svm_model = joblib.load("saved_models/svm_model.pkl")
vectorizer = joblib.load("saved_models/tfidf_vectorizer.pkl")

def predict(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    prediction = svm_model.predict(vec)[0]
    return prediction

if __name__ == "__main__":
    while True:
        user_text = input("Enter text (or 'exit'): ")
        if user_text.lower() == "exit":
            break
        print("Prediction:", predict(user_text))
