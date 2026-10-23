# Hearth & Beans Café Customer Support Chatbot ☕

## 🙂 Overview

This project is a customer support chatbot for **Hearth & Beans Café**. It uses natural language processing and a trained PyTorch neural network to identify the intent behind a customer's message and return a relevant response.

The chatbot is presented through a Streamlit web application, providing a simple chat interface for customers to ask questions about the café.

## 💡 Problem Statement & Business Impact

- Customers often need quick answers to common questions. Responding to every request manually can take time and create delays.
- **Goal:** Build a chatbot that recognizes customer message intents and responds using predefined café support answers.
- **Potential business value:** The chatbot can help answer routine questions, provide a more accessible support experience, and reduce the workload associated with repetitive enquiries.

## 📂 Project Structure

```text
Customer-Support-Department-Chatbot
│
├── data
│   └── intents.json       
│
├── model
│   └── chatbot.pth         
|
├── notebook/
│   └── Notebook.ipynb       
│
├── .gitignore              
├── app.py                  
├── LICENSE                
├── README.md            
└── requirements.txt
```

## 🧠 How It Works

1. **Intent data:** The `data/intents.json` file contains intent tags, example patterns, and responses.
2. **Text preprocessing:** messages are lowercased, punctuation is removed, english stopwords are filtered, and words are lemmatized using NLTK.
3. **Feature extraction:** A TF-IDF vectorizer converts text into numerical features.
4. **Intent prediction:** A trained PyTorch neural network predicts the intent of the user's message.
5. **Response generation:** The application selects a response associated with the predicted intent. If confidence is low, it asks the user to rephrase the question.
6. **Web interface:** Streamlit displays the conversation and maintains chat history during the session.

## 🛠️ Tech Stack

- Python
- Streamlit
- PyTorch
- NLTK
- Scikit-learn
- NumPy

## 🔮 Future Improvements

- Add more customer support intents and example questions.
- Improve intent recognition using additional training examples.
- Add a fallback flow for questions the chatbot cannot answer.
- Evaluate prediction confidence and intent-level performance.
- Deploy the application so customers can access it online.
- Add conversation analytics to identify common customer questions.