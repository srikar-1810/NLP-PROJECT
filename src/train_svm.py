# train_svm.py
# Trains a simple SVM classifier for cyberbullying detection

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

from preprocess import clean_text

def train_svm(data_path, model_out):
    df = pd.read_csv(data_path)

    # ensure correct columns
    df = df.rename(columns={"tweet_text": "text", "cyberbullying_type": "label"})
    df["clean_text"] = df["text"].apply(clean_text)

    X = df["clean_text"]
    y = df["label"]

    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42
    )

    model = LinearSVC()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("\nModel Performance:\n")
    print(classification_report(y_test, preds))

    # save model and vectorizer
    joblib.dump(model, model_out)
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

    print(f"\nSaved model to: {model_out}")
    print("Saved vectorizer to: tfidf_vectorizer.pkl")


if __name__ == "__main__":
    train_svm("../data/cleaned/cleaned_tweets.csv", "../saved_models/svm_model.pkl")
