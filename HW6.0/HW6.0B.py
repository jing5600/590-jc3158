# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 01:26:24 2021

@author: Lenovo
"""




#SOURCE: MODIFIED FROM https://blog.keras.io/building-autoencoders-in-keras.html

import keras
from keras import layers
import matplotlib.pyplot as plt

from keras.datasets import mnist,cifar10
import numpy as np

#USER PARAM
INJECT_NOISE    =   False
EPOCHS          =   20
NKEEP           =   2500        #DOWNSIZE DATASET
BATCH_SIZE      =   128
DATA            =   "MNIST"

#GET DATA
if(DATA=="MNIST"):
    (x_train, _), (x_test, _) = mnist.load_data()
    N_channels=1; PIX=28

if(DATA=="CIFAR"):
    (x_train, _), (x_test, _) = cifar10.load_data()
    N_channels=3; PIX=32
    EPOCHS=100 #OVERWRITE

#NORMALIZE AND RESHAPE
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

#ADD NOISE IF DENOISING
if(INJECT_NOISE):
    EPOCHS=2*EPOCHS
    #GENERATE NOISE
    noise_factor = 0.5
    noise= noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
    x_train=x_train+noise
    noise= noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 
    x_test=x_test+noise

    #CLIP ANY PIXELS OUTSIDE 0-1 RANGE
    x_train = np.clip(x_train, 0., 1.)
    x_test = np.clip(x_test, 0., 1.)

#BUILD CNN-AE MODEL


if(DATA=="MNIST"):
    input_img = keras.Input(shape=(PIX, PIX, N_channels))

    # #ENCODER
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)

    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
    # # AT THIS POINT THE REPRESENTATION IS (4, 4, 8) I.E. 128-DIMENSIONAL
 
    # #DECODER
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(N_channels, (3, 3), activation='sigmoid', padding='same')(x)


if(DATA=="CIFAR"):
    input_img = keras.Input(shape=(PIX, PIX, N_channels))

    #ENCODER
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    #DECODER
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(N_channels, (3, 3), activation='sigmoid', padding='same')(x)



#COMPILE
#COMPILE
autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='categorical_crossentropy',
metrics=['accuracy']);
autoencoder.summary()

#TRAIN
history = autoencoder.fit(x_train, x_train,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                shuffle=True,
                validation_data=(x_test, x_test),
                )

#HISTORY PLOT
epochs = range(1, len(history.history['loss']) + 1)
plt.figure()
plt.plot(epochs, history.history['loss'], 'bo', label='Training loss')
plt.plot(epochs, history.history['val_loss'], 'b', label='Validation loss')
plt.legend()
plt.figure()
plt.plot(epochs, history.history['accuracy'], 'bo', label='Training accuracy')
plt.plot(epochs, history.history['val_accuracy'], 'b', label='Validation accuracy')
plt.legend()

autoencoder.save("my_model3.h5")
#MAKE PREDICTIONS FOR TEST DATA
decoded_imgs = autoencoder.predict(x_test)

#VISUALIZE THE RESULTS
n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(PIX, PIX ))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(PIX, PIX ))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

