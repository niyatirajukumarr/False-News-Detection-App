{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "30975af6-53cd-41ca-8238-19a0f51b1893",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-05-01 11:23:55.846 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-01 11:23:56.326 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\NIYATI RAJUKUMAR\\AppData\\Local\\Programs\\Python\\Python39\\lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2025-05-01 11:23:56.326 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-01 11:23:56.333 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-01 11:23:56.337 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-01 11:23:56.338 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-01 11:23:56.347 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-01 11:23:56.347 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-01 11:23:56.349 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-01 11:23:56.351 Session state does not function when running a script without `streamlit run`\n",
      "2025-05-01 11:23:56.352 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-01 11:23:56.352 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-01 11:23:56.353 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-01 11:23:56.355 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-01 11:23:56.357 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-01 11:23:56.357 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-01 11:23:56.357 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import joblib\n",
    "import re\n",
    "\n",
    "# Load the model and the TF-IDF vectorizer\n",
    "model = joblib.load('news_detection_model.pkl')\n",
    "vectorizer = joblib.load('tfidf_vectorizer.pkl')\n",
    "\n",
    "# Preprocessing function (same as in the notebook)\n",
    "def preprocess_text(text):\n",
    "    text = str(text).lower()\n",
    "    text = re.sub('[^a-zA-Z\\s]', '', text)  # Remove punctuation/numbers\n",
    "    return text\n",
    "\n",
    "# Streamlit UI\n",
    "st.title(\"Fake News Detection\")\n",
    "st.write(\"This app classifies news articles as either **fake** or **real**.\")\n",
    "\n",
    "# Get the user input (news article)\n",
    "user_input = st.text_area(\"Enter News Article Here\")\n",
    "\n",
    "if st.button('Classify'):\n",
    "    # Preprocess input\n",
    "    processed_input = preprocess_text(user_input)\n",
    "    \n",
    "    # Transform the input using the loaded vectorizer\n",
    "    input_tfidf = vectorizer.transform([processed_input])\n",
    "    \n",
    "    # Make the prediction\n",
    "    prediction = model.predict(input_tfidf)\n",
    "    \n",
    "    # Display the result\n",
    "    if prediction[0] == 'fake':\n",
    "        st.error(\"The news article is **FAKE**.\")\n",
    "    else:\n",
    "        st.success(\"The news article is **REAL**.\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "149cda35-28bb-4787-bb33-8bbcc5549942",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
