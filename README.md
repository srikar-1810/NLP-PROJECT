Cyberbullying Detection Using NLP and Machine Learning

Team Members::
N.Sai Srikar[Ari079]
N.Greeshma[Ari080]
N.Likitha[Ari083]
K.Shruthika[Ari071]
G.Sahithi[Mcs072]

This project identifies different types of cyberbullying in text messages using natural language processing and machine learning techniques. The goal is to classify a given text into a specific bullying category or as non bullying.

Project Overview

The project uses a labelled cyberbullying dataset. Each text sample is assigned one type of bullying. The system cleans the text, extracts features, trains models and predicts bullying type for new inputs. The work is implemented in Python using Google Colab.

Dataset

The dataset consists of social media text labelled into the following categories:

age related bullying
ethnicity related bullying
gender related bullying
religion related bullying
not cyberbullying

The raw dataset file is stored in
data/raw/cyberbullying_tweets.csv

The cleaned dataset file is stored in
data/cleaned/cleaned_tweets.csv

Preprocessing

The preprocessing steps applied on the text include
lowercasing
removal of links
removal of non alphabet characters
removal of stopwords
lemmatization
creation of a clean text column

The preprocessing code is in
src/preprocess.py

Models
Support Vector Machine

This is the main machine learning model used. TF IDF features are extracted from the cleaned text and a LinearSVC classifier is trained. The trained model and vectorizer are saved in the saved_models folder.

Model files
saved_models/svm_model.pkl
saved_models/tfidf_vectorizer.pkl

Training script
src/train_svm.py

Prediction script
src/predict.py

LSTM Model

An LSTM based neural network is also included. This model processes sequences created from the cleaned text.

Script
src/train_lstm.py

BERT Model

A simple fine tuning script for a transformer based model is also included for experimentation.

Script
src/train_bert.py

Project Structure

data
raw
cleaned

notebooks

saved_models

src
preprocess.py
train_svm.py
predict.py
train_lstm.py
train_bert.py

How to Run

Preprocess the data
python src/preprocess.py

Train the SVM model
python src/train_svm.py

Predict using a trained model
python src/predict.py

Conclusion

The project demonstrates the use of natural language processing techniques to detect cyberbullying. The SVM model gives reliable performance for the dataset and the deep learning models provide additional experimentation options.
