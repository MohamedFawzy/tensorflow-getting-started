import  os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

#classifier using k-nearset neighbour algorithm (KNN)
# equation for distance d = squareRoot (SUM(n) where i=1 * ( x Of i - y Of i) 2)
# read mnist dataset
mnist_images = input_data.read_data_sets("MNIST_data/", one_hot=True)

# training for 100 mnist images
train_pixels , train_list_values = mnist_images.train.next_batch(100)
# test algorithm for 10 images
test_pixels, test_list_values = mnist_images.train.next_batch(10)
# define tensors for test, train
train_pixel_tensor = tf.placeholder("float", [None, 784])
test_pixel_tensor = tf.placeholder("float", [784])
# cost function
distance = tf.reduce_sum(tf.abs(tf.add(train_pixel_tensor, tf.negative(test_pixel_tensor))), reduction_indices=1)
# minimize distance
prediction = tf.arg_min(distance, 0)
# testing algorithm evaluation
accuracy = 0
init = tf.global_variables_initializer()
# start simulation
with tf.Session() as session:
    session.run(init)
    for i in range(len(test_list_values)):
        nearest_neighbour_index = session.run(prediction,
                                            feed_dict={
                                                train_pixel_tensor: train_pixels,
                                                test_pixel_tensor: test_pixels[i, :]
                                            })

        print "Test N ", i, "Predicted Class: ", np.argmax(train_list_values[nearest_neighbour_index]), "True Class: ", np.argmax(test_list_values[i])
        if np.argmax(train_list_values[nearest_neighbour_index]) == np.argmax(test_list_values[i]):
            accuracy += 1. / len(test_pixels)

print "Result = ", accuracy
