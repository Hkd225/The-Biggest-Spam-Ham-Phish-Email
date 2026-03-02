

!pip install swifter

import os
import re
import nltk
import kagglehub
import pandas as pd
import swifter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

path = kagglehub.dataset_download("akshatsharma2/the-biggest-spam-ham-phish-email-dataset-300000")

csv_path = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')][0]
df = pd.read_csv(csv_path)

df = df.dropna(subset=['text', 'label'])

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

df['cleaned_text'] = df['text'].swifter.apply(clean_text)

X = df['cleaned_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=3)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000, n_jobs=-1, class_weight='balanced')
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

import joblib

joblib.dump(model, 'email_classifier_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
