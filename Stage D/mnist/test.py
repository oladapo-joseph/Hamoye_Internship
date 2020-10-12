import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras


#the dataset
mnist = keras.datasets.mnist

(train_images, train_labels),(test_images, test_labels) = mnist.load_data()

print('Training data: '.format(train_images.shape,train_labels.shape))
