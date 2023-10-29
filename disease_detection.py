import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    return text

def preprocess_data(data_file):
    merged_data = pd.read_csv(data_file)
    merged_data['Description'].fillna('', inplace=True)
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(merged_data['Description'])
    X_train, X_test, y_train, y_test = train_test_split(
        tfidf_matrix, merged_data['Disease'], test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, tfidf_vectorizer

def train_and_predict_disease(user_input, top_n=5):
    data_file = 'Dataset\disease symptoms\merged_data.csv'
    X_train, X_test, y_train, y_test, tfidf_vectorizer = preprocess_data(data_file)

    if X_train is not None:
        model = LogisticRegression()
        model.fit(X_train, y_train)

        user_input = preprocess_text(user_input)
        user_vector = tfidf_vectorizer.transform([user_input])

        # Get predicted probabilities for all diseases
        disease_probabilities = model.predict_proba(user_vector)[0]

        # Sort diseases by probability in descending order
        sorted_indices = np.argsort(disease_probabilities)[::-1]
        sorted_diseases = [model.classes_[i] for i in sorted_indices]

        # Return the top N diseases and their probabilities
        top_n_diseases = sorted_diseases[:top_n]
        top_n_probabilities = [disease_probabilities[i] for i in sorted_indices[:top_n]]

        return top_n_diseases, top_n_probabilities
    else:
        print("Data preprocessing failed.")
        return None, None
