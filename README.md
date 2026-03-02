Project Title: Email Spam, Ham, and Phishing Classification using Machine Learning (Logistic Regression) and TF-IDF
Author: Muhammad Auffa Hakim Aditya

Description:
This project aims to classify emails into Spam, Ham, and Phishing categories using a massive dataset of ~300,000 records from Kaggle. The pipeline utilizes NLTK for text preprocessing (Stopword removal, Lemmatization) and Swifter for optimized data handling. Feature extraction is done using a TF-IDF Vectorizer (incorporating bigrams), followed by a Logistic Regression model configured with balanced class weights to handle data imbalances. The final trained model and vectorizer are exported as .pkl files for seamless deployment.

Key Technologies: Python, Scikit-Learn (TF-IDF, Logistic Regression), NLTK, Pandas, Swifter, Kagglehub, Joblib.

================================================================================

# Email Spam, Ham, and Phishing Classification
## Machine Learning with TF-IDF and Logistic Regression
### By Muhammad Auffa Hakim Aditya

This project presents a complete Natural Language Processing (NLP) pipeline for classifying emails into Spam, Ham (Normal), and Phishing categories using a large-scale dataset from Kaggle.

The project was developed by Muhammad Auffa Hakim Aditya to build an efficient Machine Learning text classification model capable of distinguishing between malicious and safe emails using English text data.

------------------------------------------------------------

PROJECT OBJECTIVES

1. Download and process ~300,000 email records from Kaggle.
2. Perform text preprocessing specialized for email content (removing links, tags, etc.).
3. Implement fast data processing using the `swifter` library.
4. Extract text features using TF-IDF Vectorization (Unigrams & Bigrams).
5. Train and evaluate a Logistic Regression model with balanced class weights.
6. Export the trained model and vectorizer for future deployment.

------------------------------------------------------------

DATASET INFORMATION

Source         : Kaggle (akshatsharma2)
Dataset Name   : The Biggest Spam Ham Phish Email Dataset
Language       : English
Total Records  : ~300,000 rows

Libraries used for data fetching:
- kagglehub
- pandas

------------------------------------------------------------

NLP PIPELINE

1. Data Cleaning
   - Drop rows with missing text or labels (NaN handling)
   - Convert all text to lowercase
   - Remove URLs (http, https, www)
   - Remove email addresses within the text
   - Remove numbers, punctuation, and special characters (keeping only a-z and spaces)

2. Stopword Removal
   - Using NLTK English stopwords to remove common non-informative words

3. Lemmatization
   - Using NLTK WordNetLemmatizer to convert words to their base dictionary form

4. Parallel Processing
   - Applied `swifter` to accelerate the pandas `apply` function during text cleaning.

------------------------------------------------------------

FEATURE ENGINEERING

TF-IDF Vectorization (Term Frequency-Inverse Document Frequency):
- max_features : 20,000 (Taking the top 20k most important vocabulary words)
- ngram_range  : (1, 2) (Capturing both single words and two-word phrases/bigrams)
- min_df       : 3 (Ignoring words that appear in fewer than 3 documents)

------------------------------------------------------------

MODEL EXPERIMENT

MODEL — Logistic Regression (Machine Learning)
- Algorithm: LogisticRegression
- Maximum Iterations: 1000
- Class Weight: 'balanced' (To handle potential dataset imbalance among Spam, Ham, and Phish classes)
- Parallel Jobs: -1 (Utilizing all processor cores)

------------------------------------------------------------

MODEL EVALUATION

The model is evaluated on a 20% stratified test split using:
- Accuracy Score
- Classification Report (Precision, Recall, and F1-Score for each class)

------------------------------------------------------------

MODEL SAVING

All trained components are saved for deployment using `joblib`:

- email_classifier_model.pkl (Trained Logistic Regression Model)
- tfidf_vectorizer.pkl (Trained TF-IDF Vectorizer)

------------------------------------------------------------

INSTALLATION

Install dependencies:
pip install pandas scikit-learn nltk swifter kagglehub joblib

Make sure to also download the required NLTK data:
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

------------------------------------------------------------

HOW TO RUN

1. Clone repository:
   git clone https://github.com/YOUR_USERNAME/email-spam-phish-classification.git

2. Install requirements.

3. Run the Python script or Jupyter Notebook. The dataset will be automatically downloaded via kagglehub.

------------------------------------------------------------

HOW TO TEST WITH NEW EMAILS (INFERENCE)

You can load the saved model and vectorizer to predict new emails without retraining:

import joblib

# Load the saved model and vectorizer
model = joblib.load('email_classifier_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Provide new email text
new_email_text = ["Congratulations! You've won a $1000 Walmart gift card. Click here to claim your prize now."]

# Vectorize and predict
email_vectorized = vectorizer.transform(new_email_text)
prediction = model.predict(email_vectorized)

print(f"The email is classified as: {prediction[0]}")

------------------------------------------------------------

AUTHOR

Muhammad Auffa Hakim Aditya

Project focus:
- Natural Language Processing (NLP)
- Cybersecurity & Email Filtering
- Machine Learning Classification
- TF-IDF Vectorization
- Text Preprocessing Optimization (Swifter)

------------------------------------------------------------

KEYWORDS 

- Muhammad Auffa Hakim Aditya
- Email Spam Classification
- Phishing Detection Machine Learning
- Kaggle Dataset NLP
- TF-IDF Logistic Regression
- Text Mining Security
- NLP Portfolio Project
