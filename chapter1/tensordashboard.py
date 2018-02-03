import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

a = tf.constant(10, name="a")
b = tf.constant(90, name="b")
y = tf.Variable(a+b * 2, name="y")

model = tf.global_variables_initializer()

with tf.Session() as session:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("/tmp/tensorflowlogs", session.graph)
    session.run(model)
    print (session.run(y))


