# tutorial example
from __future__ import absolute_import, division, print_function, unicode_literals

# import tensorflow and tf.keras
import tensorflow as tf
from tensorflow import keras

# import helper lib
import numpy as np # manager for the values
import matplotlib.pyplot as plt # display

_displayOn = False

# import the training/test sets
# ref: https://keras.io/datasets/
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# class name mapping
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

if _displayOn:
    plt.figure(1)
    plt.imshow(train_images[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()
# preprocessing
train_images, test_images = train_images / 255.0, test_images / 255.0

if _displayOn:
    plt.figure(2, figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()

# set the desired model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# compile the model
# - loss function
# - optimizer
# - metrics
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train the model
model.fit(train_images, train_labels, epochs=5)

# test the trained model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print ('\nTest accuracy:', test_acc)
