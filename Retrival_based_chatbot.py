import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class RetrivalBasedChatbot:
    def __init__(self):
        self.knowledgebase = [
            "Hello! How can I help you?",
            "Hi there! What can I do for you?",
            "Goodbye! Have a nice day.",
            "I'm a chatbot designed to assist you.",
            "I can help you with information about our services.",
            "Sorry, I don't understand that.",
            "The weather today is sunny.",
            "Our business hours are 9 AM to 5 PM.",
            "You can contact support at support@example.com.",
            "I'm not sure about that. Can you rephrase?"
        ]
        self.vectorizer = TfidfVectorizer()
        self.vector = self.vectorizer.fit_transform(self.knowledgebase)

    def get_response(self, user_input):
        user_vec = self.vectorizer.transform([user_input])

        similarity = cosine_similarity(user_vec, self.vector)

        best_match_idx = np.argmax(similarity)

        best_choice = similarity[0, best_match_idx]
        if best_choice < 0.1:
            print("Could not understand, rephrase?")
        else:
            return self.knowledgebase[best_match_idx]
if __name__=="__main__":
    bot = RetrivalBasedChatbot()
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["bye", "quit", "exit"]:
            print("bot, goodbye")
            break
        else:
            print(bot.get_response(user_input))