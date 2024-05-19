import pickle
import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import re
from sklearn.datasets import fetch_20newsgroups

# Load the dataset
newsgroups = fetch_20newsgroups(subset='all', shuffle=True, random_state=42, remove=('headers', 'footers', 'quotes'))
df = pd.DataFrame({'text': newsgroups.data, 'target': newsgroups.target})

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load trained models
nb_classifier = pickle.load(open('nb_classifier.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

def chatbot_response(user_input):
    user_input_processed = preprocess_text(user_input)
    user_input_vectorized = vectorizer.transform([user_input_processed])
    response_nb = nb_classifier.predict(user_input_vectorized)
    response_map = {i: newsgroups.target_names[i] for i in range(len(newsgroups.target_names))}
    print(response_map)
    response = f"This text belongs to the category: {response_map[response_nb[0]]}"
    return response

def main():
    st.title("Newsgroups Topic Classification Chatbot")
    st.write("Enter a sentence to classify its topic.")

    user_input = st.text_area("You: ")
    if st.button("Classify"):
        response = chatbot_response(user_input)
        st.write(response)
        st.balloons()

if __name__ == "__main__":
    main()
