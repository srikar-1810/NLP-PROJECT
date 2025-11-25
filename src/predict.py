import pickle
from preprocess import clean_sentence

# Loads the SVM model and predicts a new sentence.

def load_components():
    with open("saved_models/svm_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("saved_models/tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)

    return model, tfidf


def predict_text(sentence):
    model, tfidf = load_components()

    cleaned = clean_sentence(sentence)
    vect = tfidf.transform([cleaned])

    pred = model.predict(vect)[0]
    return pred


if __name__ == "__main__":
    test = input("Enter text: ")
    print("Prediction:", predict_text(test))
