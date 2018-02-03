
x = 1
y = x + 9
print (y)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

x = tf.constant(1, name='x')
y = tf.Variable(x+9, name='y')
print (y)
