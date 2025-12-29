import re
import random
from collections import defaultdict, Counter

class predictiveTextGenerator:
    def __init__(self):
        self.model = defaultdict(Counter)

    def preprocess_text(self,text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]','',text)
        word = text.split()
        return word
    
    def train_model(self,text):
        word = self.preprocess_text(text)
        for i in range(len(word) - 1):
            current_word = word[i]
            next_word = word[i + 1]
            self.model[current_word][next_word] += 1
    
    def predict_next_word(self, current_word):
        current_word = current_word.lower()
        if current_word in self.model:
            prediction = self.model[current_word]
            return prediction.most_common(1)[0][0]
        else:
            return "No Prediction Available"

    def add_custom_text(self, text):
        self.train_model(text)            

# ----Main Program ---- #

text_data = """
I am learning Python Programming
Python is very Easy programming Language to learn
I am working on a python intrenship Project
Machine learning (ML) is a port of Python
"""
predictor = predictiveTextGenerator()
predictor.train_model(text_data)

while True:
    print("\n1. Predict Next Word")
    print("2. Add Custom Sentence")
    print("3. Exit")

    choice = input("Enter Choice : ")

    if choice == "1":
        word = input("Enter a Word : ")
        prediction = predictor.predict_next_word(word)
        print("Suggent Next Word : ", prediction)

    elif choice == "2":
        custom_text = input("Enter Sentence to add : ")
        predictor.add_custom_text(custom_text)    
        print("Sentence Added Succesfully!")

    elif choice == "3":
        print("Exixting Program.....")
        break

    else:
        print("Invalid Choice..")        













        

