import re
import numpy as np
import joblib
import pandas as pd
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


class IntentClassificationChatbot:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words ]
        return " ".join(tokens)
    def load_train_data(self):
        training_data = {
            'text': [
                # Greeting intents - expanded dataset
                "hello", "hi there", "good morning", "hey", "greetings",
                "hello how are you", "hi good morning", "hey there", "good afternoon",
                "good evening", "howdy", "what's up", "how's it going",

                # Account balance intents - expanded
                "what is my balance", "check my account balance", "show me my balance",
                "how much money do I have", "balance inquiry", "account balance",
                "what's my current balance", "check balance please", "balance check",
                "show account balance", "current balance", "account status",

                # Money transfer intents - expanded
                "transfer money", "send money to John", "I want to transfer funds",
                "transfer $100 to my friend", "send payment", "money transfer",
                "transfer funds to account", "send money please", "wire transfer",
                "move money", "transfer cash", "send funds",

                # Bill payment intents - expanded
                "pay my electricity bill", "I want to pay bills", "bill payment",
                "pay water bill", "pay my phone bill", "bill pay",
                "I need to pay my bills", "payment for utilities", "pay rent",
                "pay credit card", "make payment", "settle bill",

                # Customer support intents - expanded
                "I need help", "customer support", "talk to agent",
                "I have a problem", "technical support", "help me please",
                "contact customer service", "I need assistance", "support",
                "help", "can you help", "I have an issue",

                # Goodbye intents - expanded
                "goodbye", "bye", "see you later", "thanks bye",
                "goodbye have a nice day", "bye bye", "see you", "farewell",
                "take care", "catch you later", "until next time", "have a good day"
            ],
            'intent': [
                # Greeting labels
                'greeting', 'greeting', 'greeting', 'greeting', 'greeting',
                'greeting', 'greeting', 'greeting', 'greeting', 'greeting',
                'greeting', 'greeting', 'greeting',

                # Balance labels
                'balance_inquiry', 'balance_inquiry', 'balance_inquiry',
                'balance_inquiry', 'balance_inquiry', 'balance_inquiry',
                'balance_inquiry', 'balance_inquiry', 'balance_inquiry',
                'balance_inquiry', 'balance_inquiry', 'balance_inquiry',

                # Transfer labels
                'money_transfer', 'money_transfer', 'money_transfer',
                'money_transfer', 'money_transfer', 'money_transfer',
                'money_transfer', 'money_transfer', 'money_transfer',
                'money_transfer', 'money_transfer', 'money_transfer',

                # Bill payment labels
                'bill_payment', 'bill_payment', 'bill_payment',
                'bill_payment', 'bill_payment', 'bill_payment',
                'bill_payment', 'bill_payment', 'bill_payment',
                'bill_payment', 'bill_payment', 'bill_payment',

                # Support labels
                'customer_support', 'customer_support', 'customer_support',
                'customer_support', 'customer_support', 'customer_support',
                'customer_support', 'customer_support', 'customer_support',
                'customer_support', 'customer_support', 'customer_support',

                # Goodbye labels
                'goodbye', 'goodbye', 'goodbye', 'goodbye',
                'goodbye', 'goodbye', 'goodbye', 'goodbye',
                'goodbye', 'goodbye', 'goodbye', 'goodbye'
            ]
        }
        return pd.DataFrame(training_data)
    def train_model(self):
        df = self.load_train_data()
        df['processed_text'] = df['text'].apply(self.preprocess_text)

        x_train, x_test,y_train, y_test = train_test_split(df['processed_text'], df['intent'], test_size=0.2, random_state=42, stratify=df['intent'])

        self.model = Pipeline([
            ("vectorizer", TfidfVectorizer(max_features=100, ngram_range=(1,2))),
            ("classifier", MultinomialNB(alpha=0.1))
        ])
        self.model.fit(x_train, y_train)

        #model evaluation

        y_pred = self.model.predict(x_test)

        print("Classification Report")
        print(classification_report(y_test, y_pred))

        #save model
        joblib.dump(self.model, "intent_classify_model.pkl")
        print("Model Trained")
        return x_test, y_test, y_pred
    def load_model(self, path="intent_classify_model.pkl"):
        try:
            self.model = joblib.load(path)
            print("model loaded successfully")
        except FileNotFoundError:
            print("file model not found, please train the model")
    def predict_intent(self, user_input, confidence_threshold=0.3):
        if self.model is None:
            return "Model not loaded", 0.0
        preprocessed_text = self.preprocess_text(user_input)

        probabilities = self.model.predict_proba([preprocessed_text])[0]
        classes = self.model.classes_

        max_prob_idx = np.argmax(probabilities)
        predicted_intent = classes[max_prob_idx]
        confidence = probabilities[max_prob_idx]

        if confidence < confidence_threshold:
            return "unclear_intent", confidence
        return predicted_intent, confidence
    def generate_response(self, intent, confidence):
        responses = {
            'greeting': [
                "Hello! How can I help you today?",
                "Hi there! What can I do for you?",
                "Good day! How may I assist you?"
            ],
            'balance_inquiry': [
                "I'll help you check your account balance. Please provide your account details.",
                "To check your balance, I'll need to verify your account information.",
                "Let me help you with your balance inquiry."
            ],
            'money_transfer': [
                "I can help you transfer money. Please provide the recipient details and amount.",
                "To process a money transfer, I'll need the destination account and amount.",
                "I'll assist you with the money transfer. What are the transfer details?"
            ],
            'bill_payment': [
                "I can help you pay your bills. Which bill would you like to pay?",
                "Let me assist you with bill payment. Please specify the bill type.",
                "I'll help you with bill payment. What bill needs to be paid?"
            ],
            'customer_support': [
                "I'll connect you with our customer support team.",
                "Let me help you with your issue. Can you describe the problem?",
                "I'm here to help. What specific assistance do you need?"
            ],
            'goodbye': [
                "Goodbye! Have a great day!",
                "Thank you for using our service. Goodbye!",
                "See you later! Take care!"
            ],
            'unclear_intent': [
                "I'm not sure what you're asking for. Could you please rephrase?",
                "I didn't quite understand. Can you be more specific?",
                "Could you please clarify what you need help with?"
            ]
        }
        import random
        return random.choice(responses.get(intent, responses['unclear_intent']))
    def chat(self):
        print("Chatbot is ready")
        print("Type 'quit' to exist the chatbot")

        while True:
            user_input = input("you: ").strip()

            if user_input =='quit':
                print("Chatbot: Goodbye")
                break
            if not user_input:
                continue
            intent, confidence = self.predict_intent(user_input)
            response = self.generate_response(intent, confidence)

            print(f"Chatbot: {response}")
            print(f"intent {intent}, confidence {confidence:.2f}")

if __name__=="__main__":
    bot = IntentClassificationChatbot()
    bot.train_model()
    bot.load_model("intent_classify_model.pkl")
    bot.chat()















