import random
import json
import pickle
import numpy as np
import nltk
from datetime import datetime as dt
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from utils.utils import SaveReports

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to load intents from a JSON file
def load_intents(filepath):
    with open(filepath) as file:
        return json.load(file)

# Preprocess data: Tokenize, lemmatize, and generate training data
def preprocess_data(intents):
    words = []
    classes = []
    documents = []
    ignore_letters = ["?", "!", ".", ","]

    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            documents.append((word_list, intent["tag"]))
            if intent["tag"] not in classes:
                classes.append(intent["tag"])

    words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
    words = sorted(set(words))
    classes = sorted(set(classes))

    training = []
    output_empty = [0] * len(classes)

    for document in documents:
        bag = []
        word_patterns = [lemmatizer.lemmatize(word.lower()) for word in document[0]]
        for word in words:
            bag.append(1 if word in word_patterns else 0)

        output_row = list(output_empty)
        output_row[classes.index(document[1])] = 1
        training.append([bag, output_row])

    random.shuffle(training)
    training = np.array(training, dtype=object)
    train_x = np.array(list(training[:, 0]))
    train_y = np.array(list(training[:, 1]))

    return words, classes, train_x, train_y

# Build and compile the model
def build_model(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(256, input_shape=(input_dim,), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim, activation='softmax'))

    sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


if __name__ == "__main__":

    intents = load_intents("intents.json")
    words, classes, train_x, train_y = preprocess_data(intents)

    pickle.dump(words, open('words.pkl', 'wb'))
    pickle.dump(classes, open('classes.pkl', 'wb'))

    model = build_model(len(train_x[0]), len(train_y[0]))
    model.fit(train_x, train_y, epochs=50, batch_size=5, verbose=1)
    model.save("model.keras")

    predictions = model.predict(train_x)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(train_y, axis=1)

    # creating an instance of SavePlot
    save = SaveReports()
    #saving classification_report.txt
    save.save_classification_report(true_classes=true_classes, predicted_classes=predicted_classes, target_names=classes)
    
    #saving cofusion matrix
    save.save_confusion_matrix(true_classes=true_classes, predicted_classes=predicted_classes,target_names=classes)
    
    #saving class distribution plot
    save.save_class_distribution(true_classes=true_classes,target_names=classes)

    # saving precision and recall plot
    save.save_precision_recall_score(true_classes=true_classes,predicted_classes=predicted_classes,target_names=classes)

