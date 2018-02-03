import tensorflow as tf
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


uniform = tf.random_uniform([100], minval=0, maxval=1, dtype=tf.float32)

sess = tf.Session()

with tf.Session() as session:
    print uniform.eval()
    plt.hist(uniform.eval(), normed=True)
    plt.show()