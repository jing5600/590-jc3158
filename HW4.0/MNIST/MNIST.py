## Organize all hyper-param in one place (top of script
NKEEP=60000
batch_size=int(0.1*NKEEP)
epochs=5
model_type = 'DFF'
data_type = 'mnist'
#CODE MODIFIED FROM:
# chollet-deep-learning-in-python

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pandas import DataFrame
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from keras.datasets import fashion_mnist
from keras.datasets import cifar10
from scipy.ndimage.interpolation import shift
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pandas import DataFrame
import warnings
from keras import models
from keras import layers
from keras.models import load_model
from keras.utils.np_utils import to_categorical
from tensorflow.keras.utils import to_categorical
##GET DATASET


import os
from keras.preprocessing import image
if data_type == 'mnist':
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
if data_type == 'fashion_mnist':
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
if data_type == 'cifar10':
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    

#QUICK INFO ON IMAGE
def get_info(image):
	print("\n------------------------")
	print("INFO")
	print("------------------------")
	print("SHAPE:",image.shape)
	print("MIN:",image.min())
	print("MAX:",image.max())
	print("TYPE:",type(image))
	print("DTYPE:",image.dtype)

#SURFACE PLOT
def surface_plot(image):
    # create the x and y coordinate arrays (here we just use pixel indices)
    xx, yy = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    fig = plt.figure()
    ax = fig.gca(projection='3d') #viridis
    ax.plot_surface(xx, yy, image[:,:] ,rstride=1, cstride=1, cmap=plt.cm.gray,linewidth=0)
    plt.show()

#CHECK NUMBER OF EACH INSTANCE
def CheckCount(data):
	S=0
	for i in range(0,10):
		count=0
		for j in range(0,len(data)):
			if(data[j]==i):
				count+=1
		print("label =",i, "    count =",count)
		S+=count
	print("TOTAL =",S)



get_info(train_images)
get_info(train_labels)
get_info(test_images)
get_info(test_labels)
get_info(train_images[0])

print("\n----- TRAINING ------")
CheckCount(train_labels)
print("\n-----   TEST   ------")
CheckCount(test_labels)
plt.imshow(train_images[0], cmap=plt.cm.gray); plt.show()
surface_plot(train_images[0])

#visualize a random image in the datase
import random
ran=random.randint(0, len(train_images))
image=train_images[ran]

from skimage.transform import rescale, resize, downscale_local_mean
image = resize(image, (10, 10), anti_aliasing=True)
print((255*image).astype(int))
get_info(image)
plt.imshow(image, cmap=plt.cm.gray); plt.show()
#exit()

print(image)
#exit()
#plt.imshow(image)
plt.imshow(image, cmap=plt.cm.gray)
plt.show()

#PRETTY PRINT THE IMAGE MATRIX 
print(DataFrame(image))




## decide the DFF OR KNN

if model_type == 'CNN':
    warnings.filterwarnings("ignore")
    
    #-------------------------------------
    #BUILD MODEL SEQUENTIALLY (LINEAR STACK)
    #-------------------------------------
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    
    model.summary()
    #-------------------------------------
    #GET DATA AND REFORMAT
    #-------------------------------------
    #from keras.utils import to_categorical
    from tensorflow.keras.utils import to_categorical
    
    train_images = train_images.reshape((60000, 28, 28, 1))
    test_images = test_images.reshape((10000, 28, 28, 1))
    
    #NORMALIZE
    train_images = train_images.astype('float32') / 255 
    test_images = test_images.astype('float32') / 255  
    
    #DEBUGGING

    print("batch_size",batch_size)
    rand_indices = np.random.permutation(train_images.shape[0])
    train_images=train_images[rand_indices[0:NKEEP],:,:]
    train_labels=train_labels[rand_indices[0:NKEEP]]
    # exit()
    
    
    #CONVERTS A CLASS VECTOR (INTEGERS) TO BINARY CLASS MATRIX.
    tmp=train_labels[0]
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    print(tmp, '-->',train_labels[0])
    print("train_labels shape:", train_labels.shape)
    
    
    history=model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size)
    
    import matplotlib.pyplot as plt
    acc = history.history['accuracy']
    loss = history.history['loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
    
    # #EVALUTE
    train_loss, train_acc = model.evaluate(train_images, train_labels, batch_size=batch_size)
    test_loss, test_acc = model.evaluate(test_images, test_labels,batch_size=test_images.shape[0])
    print('train_acc:', train_acc)
    print('test_acc:', test_acc)

        


if model_type == 'DFF':
    #INITIALIZE MODEL	
    	# Sequential model --> plain stack of layers 	
    	# each layer has exactly one input tensor and one output tensor.
    model = models.Sequential()
    
    #ADD LAYERS
    model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
    
    #SOFTMAX  --> 10 probability scores (summing to 1
    model.add(layers.Dense(10,  activation='softmax'))
    
    
    
    #PREPROCESS THE DATA
    
    #UNWRAP 28x28x MATRICES INTO LONG VECTORS (784,1) #STACK AS BATCH
    train_images = train_images.reshape((NKEEP, 28 * 28)) 
    #RESCALE INTS [0 to 255] MATRIX INTO RANGE FLOATS RANGE [0 TO 1] 
    #train_images.max()=255 for grayscale
    train_images = train_images.astype('float32') / train_images.max() 
    
    #REPEAT FOR TEST DATA
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype('float32') / test_images.max()
    

    #train_images=train_images[0:NKEEP,:,:]
    
    train_images=train_images[0:NKEEP,:]
    train_labels=train_labels[0:NKEEP]
    
    #from keras.utils import to_categorical
    from tensorflow.keras.utils import to_categorical
    
    #keras.utils.to_categorical does the following
    # 5 --> [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
    # 1 --> [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    # 9 --> [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.] ... etc
    #print(train_labels[2])
    
    train_labels = to_categorical(train_labels); #print(train_labels[2])
    test_labels = to_categorical(test_labels)
    
    ##------------------------
    #TRAIN AND EVALUTE
    ##------------------------
    
    #TRAIN

    #COMPILATION (i.e. choose optimizer, loss, and metrics to monitor)
    model.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    history=model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size)
    
    import matplotlib.pyplot as plt
    acc = history.history['accuracy']
    loss = history.history['loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
    
    # #EVALUTE
    train_loss, train_acc = model.evaluate(train_images, train_labels, batch_size=batch_size)
    test_loss, test_acc = model.evaluate(test_images, test_labels,batch_size=test_images.shape[0])
    print('train_acc:', train_acc)
    print('test_acc:', test_acc)










###Do 80-20 split of the “training” data into (train/validation)
from sklearn.model_selection import train_test_split
train_images, validation_images, train_labels, validation_labels = train_test_split(train_images, train_labels, test_size=0.2)

## AUGMENT
from keras.preprocessing import image

## Save  AND READ  MODEL
model.save("model.h5")
model = load_model("model.h5",compile=True)



## Have a function(s) that visualizes what the CNN is doing inside
img_path = 'img_1.jpg'
from keras.preprocessing import image
import numpy as np
img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.
print(img_tensor.shape)
import matplotlib.pyplot as plt
plt.imshow(img_tensor[0])
plt.show()


## Have a function(s) that visualizes what the CNN is doing inside

    
layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(test_images[0])
  
layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)  
images_per_row = 16
for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                             :, :,
                                             col * images_per_row + row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size,
                         row * size : (row + 1) * size] = channel_image
scale = 1. / size
plt.figure(figsize=(scale * display_grid.shape[1],
                    scale * display_grid.shape[0]))
plt.title(layer_name)
plt.grid(False)
plt.imshow(display_grid, aspect='auto', cmap='viridis')
