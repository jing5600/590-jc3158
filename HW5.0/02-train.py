# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 10:26:13 2021

@author: Lenovo
"""
import numpy as np
import os
os.getcwd()
os.chdir("C:\\Users\\Lenovo\\Desktop\\HW5.0")

b = np.load('processed.npz')

x_train, y_train, x_val, y_val = b['arr_0'] , b['arr_1'], b['arr_2'] ,b['arr_3']
### Training and evaluating a simple 1D convnet

from keras.models import Sequential
from keras import layers
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing import sequence
max_features = 10000
max_len = 500

model = Sequential()
model.add(layers.Embedding(max_features, 128, input_length=max_len))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))
model.summary()
model.compile(optimizer=RMSprop(lr=1e-4),
    loss='binary_crossentropy',
    metrics=['acc'])
history = model.fit(x_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2)


### Using the LSTM layer in Keras and include some form of regularization
from keras.layers import Embedding
from keras.layers import Activation, Dense
from keras.layers import LSTM
from tensorflow.python.keras import regularizers
from keras.regularizers import l2

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.compile(optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['acc'])
history = model.fit(x_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2)

###  Training the model with Embedding and SimpleRNN layers and clude some form of regularization
from keras.layers import Dense
from keras.layers import Activation, SimpleRNN
model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2)

##Plotting results and save

import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.savefig('Training and validation accuracy.png')


plt.legend()
plt.figure()



plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')


plt.savefig('Training and validation loss.png')
plt.legend()
plt.show()


## report the auc of a model
import tensorflow as tf
import keras

model.compile(optimizer='sgd',
    loss='binary_crossentropy',
    metrics=['acc'])
history = model.fit(x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(x_val, y_val))

model.save('my_model.h5')