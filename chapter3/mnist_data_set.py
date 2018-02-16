import  os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# load data set images
FLAGS = None
mnist_images = input_data.read_data_sets("MNIST_data/", one_hot=True)
pixels, real_values = mnist_images.train.next_batch(10)

print "list of values loaded ", real_values
