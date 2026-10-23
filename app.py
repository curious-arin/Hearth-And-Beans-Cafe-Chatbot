import json
import random
import re
import joblib
import streamlit as st
import torch
import torch.nn as nn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

st.set_page_config(page_title="Hearth & Beans Café", page_icon="☕")
st.title("☕ Hearth & Beans Café Chatbot")

# Load files
with open("data/intents.json", encoding="utf-8") as f:
    data = json.load(f)

vectorizer = joblib.load("model/vectorizer.pkl")
tags = joblib.load("model/tags.pkl")

# Text preprocessing (must match training)
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(lemmatizer.lemmatize(w) for w in words)

# Model architecture
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
state = torch.load(
    "model/chatbot.pth",
    map_location="cpu",
    weights_only=True
)

model = ChatbotModel(
    state["model.0.weight"].shape[1],
    state["model.0.weight"].shape[0],
    state["model.6.weight"].shape[0]
)

model.load_state_dict(state)
model.eval()

responses = {
    intent["tag"]: intent["responses"]
    for intent in data["intents"]
}

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Welcome to Hearth & Beans Café! ☕ How can I help you?"
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
if prompt := st.chat_input("Ask about our café..."):
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.write(prompt)

    features = vectorizer.transform([prompt]).toarray()
    features = torch.tensor(features, dtype=torch.float32)

    with torch.no_grad():
        output = model(features)
        probabilities = torch.softmax(output, dim=1)
        confidence, index = probabilities.max(dim=1)

    tag = tags[index.item()]

    # Debug information
    st.caption(
        f"Predicted intent: {tag} | Confidence: {confidence.item():.3f}"
    )

    if confidence.item() >= 0.15:
        reply = random.choice(
            responses.get(tag, ["Sorry, no answer found."])
        )
    else:
        reply = "Sorry, I didn't understand. Could you rephrase?"

    st.session_state.messages.append(
        {"role": "assistant", "content": reply}
    )

    with st.chat_message("assistant"):
        st.write(reply)
