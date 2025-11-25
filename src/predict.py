# predict.py
# Loads trained SVM model + TF-IDF vectorizer and predicts new text

import joblib

def load_system(model_path="svm_model.pkl", vec_path="tfidf_vectorizer.pkl"):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return model, vectorizer


def predict_text(text):
    model, vectorizer = load_system()
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    return pred


if __name__ == "__main__":
    sample1 = "I hate you so much"
    sample2 = "Have a great day friend!"

    print(sample1, "→", predict_text(sample1))
    print(sample2, "→", predict_text(sample2))
