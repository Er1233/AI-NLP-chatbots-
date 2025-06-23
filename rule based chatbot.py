import random

print("rule based chatbot")

class RuleBasedChatbot:
    def __init__(self):
        self.patterns = {
            'greeting': ['hello', 'hi', 'hey', 'good morning', 'good afternoon'],
            'goodbye': ['bye', 'goodbye', 'see you', 'farewell'],
            'name': ['what is your name', 'who are you', 'your name'],
            'help': ['help', 'assist', 'support'],
            'weather': ['weather', 'temperature', 'rain', 'sunny'],
            'time': ['time', 'clock', 'hour']
        }

        self.responses = {
            'greeting': ['Hello! How can I help you?', 'Hi there!', 'Hey! Nice to meet you!'],
            'goodbye': ['Goodbye!', 'See you later!', 'Take care!'],
            'name': ['I am ChatBot AI', 'My name is ChatBot', 'I\'m an AI assistant'],
            'help': ['I can help with greetings, weather, time, and general chat!'],
            'weather': ['I wish I could check the weather for you!', 'Sorry, I don\'t have weather data'],
            'time': ['I don\'t have access to real-time clock'],
            'default': ['I don\'t understand. Can you rephrase?', 'Interesting! Tell me more.']
        }
    def get_intent(self, user_input):
        user_input_lower = user_input.lower()
        for intent,patterns in self.patterns.items():
            for pattern in patterns:
                if pattern in user_input_lower:
                    return intent
        return "default"
    def get_response(self, user_input):
        intent = self.get_intent(user_input)
        response = self.responses.get(intent, self.responses['default'])
        return random.choice(response)
if __name__=="__main__":
    user_input = input("you: ")
    bot = RuleBasedChatbot()
    response = bot.get_response(user_input)
    print(f"Bot: {response}")