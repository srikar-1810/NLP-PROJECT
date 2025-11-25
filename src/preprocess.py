import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Cleaner designed for cyberbullying dataset
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    """Basic cleaning for tweets."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+", "", text)             # remove links
    text = re.sub(r"@\w+", "", text)                # remove mentions
    text = re.sub(r"[^a-z\s]", " ", text)           # keep alphabets only
    words = text.split()

    cleaned_words = [
        lemmatizer.lemmatize(w) for w in words if w not in stop_words
    ]

    return " ".join(cleaned_words)


def preprocess_dataset(input_path, output_path):
    """Reads original dataset and creates cleaned dataset."""
    df = pd.read_csv(input_path)

    # Rename columns to simpler names
    df = df.rename(columns={
        'tweet_text': 'text',
        'cyberbullying_type': 'label'
    })

    # Clean text column
    df['clean_text'] = df['text'].apply(clean_text)

    df.to_csv(output_path, index=False)
    print("Cleaned file saved to:", output_path)
