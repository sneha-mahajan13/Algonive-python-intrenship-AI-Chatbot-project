import json
import random
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# ---- Training Data Intents ---- #

data = {
    "intents": [
        {
            "tag" : "greetings",
            "patterns" : ["Hii", "Hello", "Hey", "Good morning"],
            "responses" : ["Hello! How can I help you?" , "Hii there! How can I assist you?"]
        },
        {
            "tag" : "order_status",
            "patterns" : ["Where is my order?", "Track my order", "Order status"],
            "responses" : ["Please provide your order Id to check the status"]
        },
        {
            "tag" : "refund",
            "patterns" : ["I want refund", "Refund my money", "Return product"],
            "responses" : ["Your refund request has been initiated"]
        },
        {
            "tag" : "Good-bye",
            "patterns" : ["Bye", "Thank you", "See you"],
            "responses" : ["Good Bye! Have a great day"]
        }
    ] 

}

# ---- Prepare Training Data ---- #

x = []
y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        x.append(pattern)
        y.append(intent["tag"])

# -- Vectorization -- # 

vectorizer = CountVectorizer()       
x_vectorizer = vectorizer.fit_transform(x)

# -- Train Model -- #
model = MultinomialNB()
model.fit(x_vectorizer, y)

# --- Chatbot Function --- #

def chatbot_response(user_input):
    input_vector = vectorizer.transform([user_input])
    tag = model.predict(input_vector)[0]

    for intent in data["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    
    return "Sorry! I didn't understand that"

# --- Console Chatbot --- #
def console_chat():
    print("AI chatbot Started (type 'exit' to stop)")
    while True:
        user = input("You : ")
        if user.lower() == "exit":
            print("Bot : GoodBye")
            break
        print("Bot : ", chatbot_response(user))    

# -- Flask Web API -- #
app = Flask(__name__)        

@app.route("/chat", methods=["POST"])
def chat_api():
    user_message = request.json.get("message")
    respomse = chatbot_response(user_message)
    return jsonify({"reply": response})

# -- Main -- #
if __name__ == "__main__":
    console_chat()    



