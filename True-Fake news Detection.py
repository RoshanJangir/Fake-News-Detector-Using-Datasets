import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# Load datasets
fake_df = pd.read_csv(r'D:\Roshan\Roshan Projects\Fake News Detector Using Datasets\True.csv')
true_df = pd.read_csv(r'D:\Roshan\Roshan Projects\Fake News Detector Using Datasets\Fake.csv')

# Add labels
fake_df['label'] = 'fake'
true_df['label'] = 'real'

# Combine datasets
df = pd.concat([fake_df, true_df], ignore_index=True)

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'[^a-z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Combine title and text
df['content'] = df['title'] + ' ' + df['text']
df['content'] = df['content'].apply(clean_text)

# Features and labels
X = df['content']
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Model pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer(
        stop_words='english',
        max_df=0.75,
        min_df=3,
        ngram_range=(1,2),
        sublinear_tf=True
    )),
    ('clf', LinearSVC())
])

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print('Accuracy:', accuracy_score(y_test, y_pred))
print('\nClassification Report:\n', classification_report(y_test, y_pred))

# User input for prediction
sample_news = input('Enter news text: ')
sample_news = clean_text(sample_news)

prediction = model.predict([sample_news])

if prediction[0] == 'fake':
    print('Prediction: Fake News')
else:
    print('Prediction: Real News')