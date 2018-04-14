# image classification problem using cnn
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# define cnn parameters
learning_rate = 0.001
training_iterations = 100000
batch_size = 128
display_step = 10
# each shape is 28 * 28 array of pixels
n_input = 784
# total classes for classifier
n_classes = 10
# dropout technique solving over fitting issues drop out units (hidden , input and output) in a neural network random
dropout = 0.75
# placeholder for the graph
x = tf.placeholder(tf.float32, [None, n_input])
# change the form of 4D input image to tensor
_X = tf.reshape(x, shape=[-1, 28, 28, 1])

# output foreach digit
y = tf.placeholder(tf.float32, [None, n_classes])

# first layer init weights and bias
wc1 = tf.Variable(tf.random_normal([5, 5, 1, 32]))
bc1 = tf.Variable(tf.random_normal([32]))

def conv2d(img , weight, bias):
    return tf.nn.relu(tf.nn.bias_add(
                tf.nn.conv2d(img, weight,
                             strides=[1, 1, 1, 1],
                             padding='SAME'), bias))


def max_pool(img , k):
    return tf.nn.max_pool(img,
                          ksize=[1, k, k, 1],
                          strides=[1, k, k, 1],
                          padding='SAME')



# convolution layer
conv1 = conv2d(_X, wc1, bc1)

conv1 = max_pool(conv1, k=2)

# reduce over fitting using dropout

keep_prob = tf.placeholder(tf.float32)
conv1 = tf.nn.dropout(conv1, keep_prob)

# second layer
wc2 = tf.Variable(tf.random_normal([5, 5, 32, 64]))
bc2 = tf.Variable(tf.random_normal([64]))

conv2 = conv2d(conv1, wc2, bc2)
conv2 = max_pool(conv2, k=2)
conv2 = tf.nn.dropout(conv2, keep_prob)

