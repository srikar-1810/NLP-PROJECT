# preprocess.py
# Basic text cleaning functions for Cyberbullying Dataset
# Written simply and clearly (human-style)

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# download resources only if they are missing
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

stop_words = set(stopwords.words("english"))
lemm = WordNetLemmatizer()

def clean_text(text):
    """Cleans a single text string by removing links, symbols and stopwords."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+", "", text)            # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)          # keep only alphabets
    text = re.sub(r"\s+", " ", text).strip()       # remove extra spaces

    words = []
    for w in text.split():
        if w not in stop_words:
            words.append(lemm.lemmatize(w))

    return " ".join(words)


def apply_cleaning(df):
    """Applies cleaning to the dataframe that contains a 'text' column."""
    df["clean_text"] = df["text"].apply(clean_text)
    return df
