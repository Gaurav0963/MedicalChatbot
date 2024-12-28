import os
import glob
import subprocess
import pickle
import numpy as np
import nltk
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer
import json
import random

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents.json file
def load_intents(filepath="intents.json"):
    with open(filepath, "r") as file:
        return json.load(file)

# Preprocess user input
def preprocess_input(user_input, words):
    tokens = nltk.word_tokenize(user_input)
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens]
    bag = [1 if word in tokens else 0 for word in words]
    return np.array(bag)

# Predict intent
def predict_intent(user_input, model, words, classes):
    bow = preprocess_input(user_input, words)
    result = model.predict(np.array([bow]))[0]
    confidence_threshold = 0.5
    if max(result) > confidence_threshold:
        return classes[np.argmax(result)]
    return "unknown"

# Get response for intent
def get_response(intent, intents_json):
    for intent_data in intents_json["intents"]:
        if intent_data["tag"] == intent:
            return random.choice(intent_data["responses"])
    return "I'm sorry, I didn't understand that."

# Start the chatbot
def chatbot(model, words, classes, intents):
    print("Chatbot is ready! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("Chatbot: Goodbye!")
            break
        intent = predict_intent(user_input, model, words, classes)
        response = get_response(intent, intents)
        print(f"Chatbot: {response}")

# Main function
def main():
    # Check if .keras file exists
    model_file = glob.glob("*.keras")
    if model_file:
        print(f"Found model: {model_file[0]}")
        model = load_model(model_file[0])
    else:
        print("No model found. Building model by running build_model.py...")
        # Run build_model.py to create the model
        subprocess.run(["python", "build_model.py"], check=True)
        # Reload the model after creation
        model_file = glob.glob("*.keras")
        if model_file:
            model = load_model(model_file[0])
        else:
            print("Failed to create the model. Exiting...")
            return

    # Load necessary files
    words = pickle.load(open("words.pkl", "rb"))
    classes = pickle.load(open("classes.pkl", "rb"))
    intents = load_intents("intents.json")

    # Start the chatbot
    chatbot(model, words, classes, intents)

if __name__ == "__main__":
    main()
