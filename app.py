import streamlit as st
import joblib
import re

import requests

# Function to verify using NewsAPI
def search_news(query):
    api_key = 'YOUR_NEWSAPI_KEY'  # Replace this with your actual API key
    url = f'https://newsapi.org/v2/everything?q="{query}"&language=en&sortBy=relevancy&apiKey={api_key}'
    
    try:
        response = requests.get(url)
        data = response.json()
        if data["status"] == "ok" and data["totalResults"] > 0:
            return data["articles"]
        else:
            return []
    except Exception as e:
        st.warning(f"News API error: {e}")
        return []
    
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
    
    # Transform input using vectorizer
    input_tfidf = vectorizer.transform([processed_input])
    
    # Make ML prediction
    prediction = model.predict(input_tfidf)
    
    # Check via NewsAPI
    search_results = search_news(user_input)

    # Show model prediction
    if prediction[0] == 'fake':
        st.error("The news article is **FAKE** according to the ML model.")
    else:
        st.success("The news article is **REAL** according to the ML model.")

    # Show NewsAPI results
    st.markdown("---")
    st.subheader("🔎 Real-Time News Verification")

    if search_results:
        st.success("✅ Similar news found on trusted sources:")
        for article in search_results[:3]:
            st.write(f"- [{article['title']}]({article['url']}) ({article['source']['name']})")
    else:
        st.warning("❌ No similar news found in real-time trusted sources.")
