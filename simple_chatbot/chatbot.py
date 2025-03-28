import random
import numpy as np
import spacy
import json

with open('./data/json file/intents.json') as f:
    data = json.load(f)

nlp = spacy.load('en_core_web_sm')
words = []
labels = []
docs_x = []
docs_y = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        for token in nlp(pattern):      # tokenizing every pattern (sentence from database)
             words.append(token)        # keeping tokens in words list
        docs_x.append(nlp(pattern))     # keeping sentences in docs_x list
        docs_y.append(intent['tag'])    # keeping tags, e.g. labels, for each sentence in docs_y list

    if intent['tag'] not in labels:
        labels.append(intent['tag'])

words = [w.lemma_ for w in words if w.lemma_ not in ['.', '?', '!', ',']] # Lemmatization of elements in list words
words = sorted(list(set(words))) # In order to apply bag of words method, words list needs to have unique elements
labels = sorted(labels)

training = []
output = []
for x, sentence in enumerate(docs_x):
    bag = np.zeros((1, len(words)))

    words_in_sentence = [w.lemma_ for w in sentence if w.lemma_ not in ['.', '?', '!', ',']]
    indexes = []
    for elem1 in words_in_sentence:
        for elem2 in words:
            if elem1 == elem2:
                bag[0, words.index(elem2)] = 1  # Bag of words

    output_prep = np.zeros((1, len(labels)))
    output_prep[0, labels.index(docs_y[x])] = 1  # Output is one-hot encoded

    training.append(bag)
    output.append(output_prep)

training = np.array(training)
output = np.array(output)

#### MODEL ####

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.losses import CategoricalCrossentropy
from keras import regularizers

model = Sequential([
    Dense(512, input_shape=(1, training.shape[2]), activation='relu'),
    Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dense(64, activation='relu'),
    Dense(len(labels), activation='softmax')
])

es = EarlyStopping(monitor='val_accuracy', patience=50)
model.compile(optimizer=Adam(0.001), loss=CategoricalCrossentropy(), metrics='accuracy')
model.fit(training, output, epochs=2500, callbacks=[es], validation_split=0.3)

#### IMPLEMENTING USER CHAT ####

while True:
    user_input = input('Type your message: ')
    if user_input.lower() in ['finish', 'quit', 'exit']: # Break from loop
        break

    tokens = nlp(user_input)
    words_in_sentence = [w.lemma_ for w in tokens  if w.lemma_ not in ['.', '?', '!', ',']]

    user_input_one_hot = np.zeros((1, len(words)))
    for token in words_in_sentence:
        for word in words:
            if token == word:
                user_input_one_hot[0, words.index(word)] = 1

    prediction = model.predict(np.array([user_input_one_hot]))

    id_label = np.argmax(prediction[0], axis=1)
    target_label = labels[id_label[0]]
    id_target = -1
    for i in range(len(data['intents'])):
        if data['intents'][i]['tag'] == target_label:
            id_target = i

    k = random.choice(data['intents'][id_target]['responses']) # Choosing random answer from database
    print('Response: ', k)