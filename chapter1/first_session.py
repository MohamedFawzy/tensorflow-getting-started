import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

x = tf.constant(1, name='x')
y = tf.Variable(x+9, name='y')

model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)
    print (session.run(y))

a = tf.placeholder("int32")
b = tf.placeholder("int32")

y = tf.multiply(a, b)

session = tf.Session()

print session.run(y, feed_dict={a: 2, b: 5})
