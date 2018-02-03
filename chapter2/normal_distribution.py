import tensorflow as tf
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


norm = tf.random_normal([100], mean=0, stddev=2)

with tf.Session() as session:
    plt.hist(norm.eval(), normed=True)
    plt.show()