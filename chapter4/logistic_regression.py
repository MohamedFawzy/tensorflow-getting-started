import  os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

# logistic regression algorithm using neural network
# problem : classify images from mnist data set

# load data set images
mnist_images = input_data.read_data_sets("MNIST_data/", one_hot=True)
