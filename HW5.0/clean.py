# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 08:13:04 2021

@author: Lenovo
"""

import os
os.getcwd()
os.chdir("C:\\Users\\Lenovo\\Desktop\\HW5.0")
import re


## for file Pride and Prejudice
open_diff = open('Pride and Prejudice.txt', 'r', encoding='UTF-8')  
diff_line = open_diff.readlines()

line_list = []
for line in diff_line:
    line_list.append(line)

count = len(line_list) 
print('the count of lines of original dataset',count)

diff_match_split = [line_list[i:i+300] for i in range(0,len(line_list),300)]#  

for i,j in zip(range(0,int(count/300+1)),range(0,int(count/300+1))): #  
    with open('./train/Pride and Prejudice/Pride and Prejudice%d.txt'% j,'w+', encoding='UTF-8') as temp:
        for line in diff_match_split[i]:
            temp.write(line)
print('the count of files after splited',i+1)


## for file The Scarlet Letter
open_diff = open('The Scarlet Letter.txt', 'r', encoding='UTF-8') #  
diff_line = open_diff.readlines()

line_list = []
for line in diff_line:
    line_list.append(line)

count = len(line_list) 
print('the count of lines of original dataset',count)

diff_match_split = [line_list[i:i+300] for i in range(0,len(line_list),300)]# 

for i,j in zip(range(0,int(count/300+1)),range(0,int(count/300+1))): #  
    with open('./train/The Scarlet Letter/The Scarlet Letter%d.txt'% j,'w+', encoding='UTF-8') as temp:
        for line in diff_match_split[i]:
            temp.write(line)
print('the count of files after splited',i+1)


## for the file Dracula
open_diff = open('Dracula.txt', 'r', encoding='UTF-8') #  
diff_line = open_diff.readlines()

line_list = []
for line in diff_line:
    line_list.append(line)

count = len(line_list) 
print('the count of lines of original dataset',count)

diff_match_split = [line_list[i:i+300] for i in range(0,len(line_list),300)]#  

for i,j in zip(range(0,int(count/300+1)),range(0,int(count/300+1))): #  
    with open('./train/Dracula/Dracula%d.txt'% j,'w+', encoding='UTF-8') as temp:
        for line in diff_match_split[i]:
            temp.write(line)
print('the count of files after splited',i+1)



###  Processing the labels

import os
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
        
### Tokenizing
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
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
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]


## save the array
import numpy as np

np.savez_compressed('processed.npz',  x_train, y_train, x_val, y_val)

