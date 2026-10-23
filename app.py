
import streamlit as st
import torch
import torch.nn as nn
import joblib
import json
import re
import random
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load trained files
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vectorizer = joblib.load("model/vectorizer.pkl")
tags = joblib.load("model/tags.pkl")

with open("data/intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(lemmatizer.lemmatize(w) for w in words)

# Match the model architecture in your notebook
class ChatbotModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.02),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.02),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x)

@st.cache_resource
def load_model():
    model = ChatbotModel(
        len(vectorizer.get_feature_names_out()), 16, len(tags)
    ).to(device)
    model.load_state_dict(
        torch.load("model/chatbot.pth", map_location=device, weights_only=True)
    )
    model.eval()
    return model

model = load_model()

def get_response(message):
    text = vectorizer.transform([preprocess(message)]).toarray()
    inputs = torch.tensor(text, dtype=torch.float32).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(inputs), dim=1)
        confidence, prediction = torch.max(probabilities, dim=1)

    if confidence.item() < 0.35:
        return "Sorry, I didn't understand. Could you ask about our café menu or services?"

    tag = tags[prediction.item()]
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "Welcome to Hearth & Beans Café! How can I help you?"

# Streamlit interface
st.set_page_config(page_title="Hearth & Beans Café", page_icon="☕")
st.title("☕ Hearth & Beans Café")
st.caption("Your cozy corner for coffee, treats, and good conversations.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Hello! Welcome to Hearth & Beans Café. How can I help you today?"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask about our café..."):
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )
    with st.chat_message("user"):
        st.write(prompt)

    reply = get_response(prompt)
    st.session_state.messages.append(
        {"role": "assistant", "content": reply}
    )
    with st.chat_message("assistant"):
        st.write(reply)
