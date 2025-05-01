import streamlit as st
import joblib
import re

# Load the model and the TF-IDF vectorizer
model = joblib.load('news_detection_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Preprocessing function (same as in the notebook)
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub('[^a-zA-Z\s]', '', text)  # Remove punctuation/numbers
    return text

# Streamlit UI
st.title("Fake News Detection")
st.write("This app classifies news articles as either **fake** or **real**.")

# Get the user input (news article)
user_input = st.text_area("Enter News Article Here")

if st.button('Classify'):
    # Preprocess input
    processed_input = preprocess_text(user_input)
    
    # Transform the input using the loaded vectorizer
    input_tfidf = vectorizer.transform([processed_input])
    
    # Make the prediction
    prediction = model.predict(input_tfidf)
    
    # Display the result
    if prediction[0] == 'fake':
        st.error("The news article is **FAKE**.")
    else:
        st.success("The news article is **REAL**.")
