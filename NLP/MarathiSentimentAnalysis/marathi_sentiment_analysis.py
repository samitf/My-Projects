import streamlit as st
import re
import nltk
import joblib

# Load the model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

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

# Define a function for predicting sentiment
def predict_sentiment(text):
    cleaned_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)
    return prediction[0]

# Streamlit interface
st.title('Marathi Sentiment Analysis')
st.write('Enter a Marathi sentence and get its sentiment label.')

user_input = st.text_input('Enter Sentence:', '')

# Button for submission
if st.button("Submit"):
    if user_input:
        sentiment = predict_sentiment(user_input)
        if sentiment == 1:
            sentiment_label = "Positive"
        elif sentiment == -1:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"

        st.write(f'Predicted Sentiment: {sentiment_label}')
    else:
        st.write("Please enter a sentence.")
