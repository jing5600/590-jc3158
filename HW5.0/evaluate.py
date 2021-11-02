# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 11:17:34 2021

@author: Lenovo
"""
import numpy as np
import os
os.getcwd()
os.chdir("C:\\Users\\Lenovo\\Desktop\\HW5.0")
import os
from keras.preprocessing.text import Tokenizer
import nltk
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import keras
from keras import models
from keras.models import load_model

maxlen = 100

ori_dir = "C:\\Users\\Lenovo\\Desktop\\HW5.0"
train_dir = os.path.join(ori_dir, 'train')
labels = []
texts = []
for label_type in ['Dracula', 'The Scarlet Letter','Pride and Prejudice']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname), encoding='UTF-8')
            texts.append(f.read())
            f.close()
            if label_type == 'Dracula':
                labels.append(0)
            if label_type == 'The Scarlet Letter':
                labels.append(0)
            else:
                labels.append(1)
                

maxlen = 100
training_samples = 200
validation_samples = 10000
max_words = 10000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=maxlen)
labels = np.asarray(labels)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
x_test = data[:training_samples]
y_test = labels[:training_samples]



model=load_model('my_model.h5')
model.evaluate(x_test, y_test)
