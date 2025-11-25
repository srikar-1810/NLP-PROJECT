import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from preprocess import clean_dataframe

# This file trains a simple SVM model on the cleaned data.
# SVM works well for text tasks and is quick to train.

def train_svm():
    df = pd.read_csv("data/raw/cyberbullying_data.csv")
    df = clean_dataframe(df, "text")

    tfidf = TfidfVectorizer(max_features=5000)
    x = tfidf.fit_transform(df["text"])
    y = df["label"]

    model = LinearSVC()
    model.fit(x, y)

    # Save both model and vectorizer for later prediction
    import pickle
    with open("saved_models/svm_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("saved_models/tfidf.pkl", "wb") as f:
        pickle.dump(tfidf, f)

    print("SVM training complete. Model saved.")

if __name__ == "__main__":
    train_svm()
