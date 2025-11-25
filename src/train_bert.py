# train_bert.py
# Light BERT fine-tuning example (minimal)

import pandas as pd
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

def train_bert(data_path, save_path):
    df = pd.read_csv(data_path)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    X = list(df["text"])
    y = pd.factorize(df["label"])[0]

    enc = tokenizer(X, truncation=True, padding=True, max_length=70)

    model = TFBertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=len(set(y))
    )

    history = model.fit(
        dict(enc),
        y,
        epochs=1,
        batch_size=6
    )

    model.save_pretrained(save_path)
    print("BERT model saved:", save_path)
