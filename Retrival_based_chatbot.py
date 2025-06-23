from sklearn.feature_extraction.text import TfidfVectorizer

import nltk


print("Retrival-based-chatbot")

class RetrivalBased:
    def __init__(self):
        self.knowledge_base = [
            "Hello! How can I help you today?",
            "I'm an AI chatbot designed to assist you.",
            "I can answer questions about various topics.",
            "What would you like to know?",
            "I'm here to help with information and support.",
            "Feel free to ask me anything!",
            "I use natural language processing to understand you.",
            "My goal is to provide helpful responses.",
            "I'm constantly learning to improve my responses.",
            "Thank you for chatting with me!"
        ]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.vectors= self.vectorizer.fit_transform(self.knowledge_base)


