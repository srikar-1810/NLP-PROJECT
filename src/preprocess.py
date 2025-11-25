import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_sentence(text):
    """
    Cleans a single sentence by removing unwanted characters,
    converting to lowercase and applying lemmatization.
    """

    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+", "", text)         
    text = re.sub(r"[^a-z\s]", "", text)         
    words = text.split()

    cleaned = []
    for w in words:
        if w not in stop_words:
            cleaned.append(lemmatizer.lemmatize(w))

    return " ".join(cleaned)


def clean_dataframe(df, text_column):
    """
    Applies the clean_sentence function to every row of the dataset.
    """

    df[text_column] = df[text_column].apply(clean_sentence)
    return df
