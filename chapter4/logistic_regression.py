import  os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# logistic regression algorithm using neural network
# problem : classify images from mnist data set

# load data set images
mnist_images = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1
# tf graph input
x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float",  [None, 10])
# create model
# set model weights
W = tf.Variable(tf.zeros[784, 10])
# set model bias
b = tf.Variable(tf.zeros[10])
# construct model
activation = tf.nn.softmax(tf.multiply(x, W)+ b)
# minimize error using cross entropy
cross_entropy = y * tf.log(activation)
cost = tf.reduce_mean(tf.reduce_sum(cross_entropy, reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# plot setting
avg_set = []
epoch_set = []
#init variables
init = tf.global_variables_initializer()
# launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = init(mnist_images.train.num_examples/batch_size)
        # loop over all batches
        for i in range(total_batch):
            batches_xs, batches_ys = mnist_images.train.next_batch(batch_size)
            # fit training using batch data
            sess.run(optimizer, feed_dict={x: batches_xs, y: batches_ys})
            # compute average loss
            avg_cost += sess.run(cost, feed_dict={x: batches_ys, y:batches_ys})/total_batch
            # Display logs per epoch step
            if epoch % display_step==0:
                print "Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost)
                avg_set.append(avg_cost)
                epoch_set.append(epoch+1)
                print "Training phase finished"
                plt.plot(epoch_set, avg_set, 'o',label='Logistic Regression Training phase')
                plt.ylabel('cost')
                plt.xlabel('epoch')
                plt.legend()
                plt.show()
 # Test model
correct_prediction = tf.equal(tf.argmax(activation, 1),tf.argmax(y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print "Model accuracy:", accuracy.eval({x: mnist_images.test.images,y: mnist_images.test.labels})