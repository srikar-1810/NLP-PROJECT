# train_lstm.py
# Simple LSTM model for text classification (not too complex)

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from preprocess import clean_text

def train_lstm(data_path, save_path):
    df = pd.read_csv(data_path)
    df["clean_text"] = df["text"].apply(clean_text)

    texts = df["clean_text"].tolist()
    labels = pd.factorize(df["label"])[0]

    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(texts)

    seq = tokenizer.texts_to_sequences(texts)
    seq = pad_sequences(seq, maxlen=40)

    model = Sequential([
        Embedding(5000, 64),
        LSTM(64),
        Dense(32, activation="relu"),
        Dense(len(set(labels)), activation="softmax")
    ])

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])

    model.fit(seq, labels, epochs=3, batch_size=32)

    model.save(save_path)
    print("LSTM model saved to:", save_path)
