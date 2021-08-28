#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# -------------- Importing the libraries ------------------- 

import nltk
# CAMeL Tools is suite of Arabic natural language processing tools
# developed by the CAMeL Lab at New York University Abu Dhabi.
#from nltk.stem import WordNetLemmatizer
from nltk.stem.isri import ISRIStemmer #arabic stemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from camel_tools.tokenizers.word import simple_word_tokenize #arabic tokenizer
from camel_tools.utils.dediac import dediac_ar
from camel_tools.utils.normalize import normalize_alef_maksura_ar
from camel_tools.utils.normalize import normalize_alef_ar
from camel_tools.utils.normalize import normalize_teh_marbuta_ar
import json
import pickle
import numpy as np
import random


# importing Keras libraries
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

# -------------- load the dataset -------------------

data_file = open('dataset.json').read()
dataset = json.loads(data_file)

# -------------- Cleaning the text -------------------
#                preprocess data


lemmatizer = WordNetLemmatizer()

# create a list of all the words in the file 
words=[]
#create a list of classes for our tags
classes = []
# craate teh corpus after cleaning the text
corpus = []
ignore_chars = ['؟', '!','،',',','.','.','?']



# loop through the dataset.json file
for data in dataset['intents']:
    for pattern in data['patterns']:
    # remove stopwords with a for loop
    # after cleaining we'll add it to corpus    
    
        #tokenize each statement into words
        w = nltk.word_tokenize(pattern)
        # add the tokenize words to words[]
        words.extend(w)
        #add each tokenized words to its tag e.g., ['أهلا','مرحبا'] greeting
        corpus.append((w, data['tag']))
        
        # add all the tags to our classes list
        if data['tag'] not in classes:
            classes.append(data['tag'])

#return the root of the word        
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_chars]


#------------------


#Arabic_stop_words = list(stopwords.words("arabic"))


#for w in words:
 #   if w in Arabic_stop_words:
  #      words.remove(w)
   # # Normalize alef variants to 'ا'
    #nw = normalize_alef_ar(w)
    # Normalize alef maksura 'ى' to yeh 'ي'
  #  nw = normalize_alef_maksura_ar(w)
   # # Normalize teh marbuta 'ة' to heh 'ه'
    #nw = normalize_teh_marbuta_ar(w)
    # removing Arabic diacritical marks
    #nw = dediac_ar(w)
    #words.extend(nw)

stemmer = ISRIStemmer() 

#words = [stemmer.stem(w) for w in words if w not in ignore_chars]

Arabic_stop_words= list(stopwords.words("Arabic")) 


[words.remove(w) for w in words if w in Arabic_stop_words] 


#---------------------------

# sort words
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))

#              #  print #               #

# print corpus size 
print (len(corpus), "corpus")
# print classes size with classes = intents
print (len(classes), "classes", classes)
# print words size = all lemmatized words, vocabulary
print (len(words), "unique lemmatized words", words)

# to save time instead of biulding them each time we open the GUI
pickle.dump(words,open('normalized_words.pkl','wb'))
pickle.dump(classes,open('classes_2.pkl','wb'))

# -------------- Create training and testing data ------------------- 
#                 creating the Bag of Words model

# Splitting the dataset into the Training set and Test set
# input = the pattern, output = the class our input pattern belongs to 
# But the computer doesn’t understand text, so we will convert text into numbers:


# create the training data
training = []
# create an empty array for the output that matches the size of classes tags
array_empty = [0] * len(classes)

# training set, bag of words for each sentence
for cor in corpus: # tokenized words with thier tags: ['أهلا','مرحبا']greeting
    # initialize the bag of words 
    bag = [] 
    # list of tokenized words for the pattern [0]['أهلا','مرحبا'] 
    pattern_words = cor[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(w.lower()) for w in pattern_words]
    
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(array_empty)
    output_row[classes.index(cor[1])] = 1 # cor[1] = greeting
    
    # create the training data(0's and 1's) with the [bag of words][classes]
    training.append([bag, output_row]) 
    
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
# create train and test lists. X - patterns, Y - intents, classes
train_x = list(training[:,0]) # all rows, first col 
train_y = list(training[:,1]) # all rows, 2nd col, which pattern belongs to which class
print("Training data created")


# -------------- Bilding the model -------------------



# Define model architecture, linear stack of layers
model = Sequential()

# Create the model
# 3 layers. First layer 128 neurons, second layer 64 neurons 
# 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu')) #train_x = size
model.add(Dropout(0.5)) # to control overfitting problem
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax')) # which pattern belongs to which class

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) #learning rate =0.01
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Fitting and saving the model 
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_arabic_model.h5', hist) # to save time instead of training the model each time
print("model created")