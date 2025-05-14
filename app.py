import streamlit as st
import pickle
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK stopwords (only runs once)
nltk.download('stopwords')

# Load the saved model and vectorizer
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

# Preprocessing function
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Streamlit UI
st.title("📰 Fake News Detector")
st.write("Enter a news article below to check if it's **Fake** or **True**.")

user_input = st.text_area("Paste news article text here", height=200)

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        clean_text = preprocess_text(user_input)
        vector = vectorizer.transform([clean_text])
        st.write("Cleaned text:", clean_text)
        st.write("Vector shape:", vector.shape)
        st.write("Vector nonzero elements:", vector.nnz)
        proba = model.predict_proba(vector)[0]
        st.write(f"Prediction probabilities - Fake: {proba[0]:.2f}, True: {proba[1]:.2f}")
        prediction = model.predict(vector)[0]
        label = "Fake News" if prediction == 0 else "True News"
        st.success(f"Prediction: **{label}**")
