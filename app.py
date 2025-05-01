import streamlit as st
import re
import requests
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax

# ---------------------
# Load BERT model and tokenizer
# ---------------------
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained('mrm8488/bert-tiny-finetuned-fake-news')
    model = BertForSequenceClassification.from_pretrained('mrm8488/bert-tiny-finetuned-fake-news', num_labels=2)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ---------------------
# NewsAPI search function
# ---------------------
def search_news(query):
    api_key = 'b4f89375b21c4664adfc739250d7c042'  # Replace with your actual key
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

# ---------------------
# Predict using BERT
# ---------------------
def predict_bert(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=1)
        confidence = probs[0].tolist()
        label = 'real' if torch.argmax(outputs.logits) == 1 else 'fake'
    return label, confidence

# ---------------------
# Streamlit UI
# ---------------------
st.title("🕵️ Fake News Detection with BERT + Real-Time Verification")
st.write("This app uses a BERT model and trusted news sources to classify news as **fake** or **real**.")

user_input = st.text_area("📰 Enter News Article Here")

if st.button('Classify'):
    if not user_input.strip():
        st.warning("Please enter a news article.")
    else:
        with st.spinner("Analyzing with BERT..."):
            label, confidence = predict_bert(user_input)
            search_results = search_news(user_input)

        # NewsAPI Verification First
        st.markdown("---")
        st.subheader("🔍 Real-Time NewsAPI Verification")

        if search_results:
            st.success("✅ Similar articles found on trusted sources. Likely **REAL**.")
            for article in search_results[:3]:
                st.markdown(f"- [{article['title']}]({article['url']}) ({article['source']['name']})")
        else:
            # If not found in trusted sources, fallback to BERT result
            st.subheader("🤖 BERT Model Prediction")
            if label == 'fake':
                st.error(f"⚠️ **FAKE NEWS** detected by BERT (Confidence: {round(confidence[0]*100, 2)}%)")
            else:
                st.success(f"✅ **REAL NEWS** detected by BERT (Confidence: {round(confidence[1]*100, 2)}%)")

            st.info("🤔 ML model thinks it's REAL, but couldn't verify online.")
