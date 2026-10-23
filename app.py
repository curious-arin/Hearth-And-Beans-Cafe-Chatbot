import json
import random
import re
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Files
data_path = Path("data/intents.json")
model_path = Path("model/chatbot.pth")

st.set_page_config(page_title="Hearth & Beans Café", page_icon="☕")
st.title("☕ Hearth & Beans Café Chatbot")

# Load intents
with open(data_path, encoding="utf-8") as f:
    data = json.load(f)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(lemmatizer.lemmatize(w) for w in words)

# Rebuild TF-IDF from training patterns
patterns, tags = [], []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(preprocess(pattern))
        tags.append(intent["tag"])

unique_tags = sorted(set(tags))
vectorizer = TfidfVectorizer()
vectorizer.fit(patterns)

# Model
class ChatbotModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.005),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.005),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x)

# Load trained model
state = torch.load(model_path, map_location="cpu", weights_only=True)
input_size = state["model.0.weight"].shape[1]
hidden_size = state["model.0.weight"].shape[0]
output_size = state["model.6.weight"].shape[0]

if len(vectorizer.vocabulary_) != input_size or len(unique_tags) != output_size:
    st.error("intents.json does not match the trained chatbot model.")
    st.stop()

model = ChatbotModel(input_size, hidden_size, output_size)
model.load_state_dict(state)
model.eval()

responses = {i["tag"]: i["responses"] for i in data["intents"]}

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome to Hearth & Beans Café! ☕"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt = st.chat_input("Ask about our café...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    features = vectorizer.transform([preprocess(prompt)]).toarray()
    features = torch.tensor(features, dtype=torch.float32)

    with torch.no_grad():
        output = model(features)
        probabilities = torch.softmax(output, dim=1)
        confidence, index = probabilities.max(dim=1)

    tag = unique_tags[index.item()]

    if confidence.item() < 0.15:
        reply = "Sorry, I didn't understand. Could you rephrase?"
    else:
        reply = random.choice(responses.get(tag, ["Sorry, no answer found."]))

    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.write(reply)