import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

train_df = pd.read_csv('train.csv', encoding='utf-8')
val_df = pd.read_csv('valid.csv', encoding='utf-8')
test_df = pd.read_csv('test.csv', encoding='utf-8')

def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        stopwords = f.read().splitlines()
    return set(stopwords)

marathi_stopwords = load_stopwords('marathi_stopwords.txt')


def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    tokens = nltk.word_tokenize(text)  # Tokenization
    tokens = [word for word in tokens if word not in marathi_stopwords]  # Remove stopwords
    return ' '.join(tokens)

# Apply preprocessing to each dataset
train_df['cleaned_text'] = train_df['tweet'].apply(preprocess_text)
val_df['cleaned_text'] = val_df['tweet'].apply(preprocess_text)
test_df['cleaned_text'] = test_df['tweet'].apply(preprocess_text)

# Initialize CountVectorizer
vectorizer = CountVectorizer()

# Vectorize training data
X_train_vectorized = vectorizer.fit_transform(train_df['cleaned_text'])
y_train = train_df['label']

# Initialize and train the model
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Save the trained model and vectorizer
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Model and vectorizer saved successfully.")
