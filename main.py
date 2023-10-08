import tensorflow as tf             # DL library
import numpy as np                  # lin alg library
import matplotlib.pyplot as plt     # visualization
import cv2
import os
import math

def step(x):
    return tf.math.floor(x) + 1

def square(x):
    return tf.keras.backend.switch(x >= 0, tf.math.square(x), tf.math.floor(x) + 1)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() # grab dataset
# turn input into unit vectors
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential() # set up NN
model.add(tf.keras.layers.Flatten(input_shape=(28, 28))) # add input layer (flatten means force everything into 1 x n vector)
model.add(tf.keras.layers.Dense(784, activation='tanh')) # layer where all input notes are connected to every node in the dense layer
# model.add(tf.keras.layers.Dense(784, activation='relu')) # another relu for more computations? not well explained
# model.add(tf.keras.layers.Dense(64, activation='relu')) 
model.add(tf.keras.layers.Dense(10, activation='softmax'))  # output layer for 10 digits -- softmax takes the max of the 10 neurons

# set the training settings of NN 
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.00025), loss='sparse_categorical_crossentropy', metrics=['accuracy']) # metrics = stats to print
# optimization algorithms: 
# adam is a gradient descent algorithm that takes into account second-order moments (99.5% acc)
# FTRL = shallow models wiht large features <-- really bad for digit problem (11% acc)
# lion is another gradient descent optmizer, but is more momentum based, works better with large datasets (99.2% acc)
# SGD is normal gradient descent algoritm (95% acc)

model.fit(x_train, y_train, epochs=10)

model.save("KNN-10-epochs-slower2-learning-more1-nodes-layer1-tanh-activator.model")