import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

uniform_with_seed = tf.random_uniform([1], seed=1)
uniform_without_seed = tf.random_uniform([1])

print ("First Run")
with tf.Session() as first_session:
    print ("uniform with (seed = 1) = {}").format(first_session.run(uniform_with_seed))
    print ("uniform with (seed = 1) = {}").format(first_session.run(uniform_with_seed))
    print ("uniform without seed = {}").format(first_session.run(uniform_without_seed))
    print ("uniform without seed = {}").format(first_session.run(uniform_without_seed))

print ("Second Run")
with tf.Session() as second_session:
    print ("uniform with (seed=1) = {}").format(second_session.run(uniform_with_seed))
    print ("uniform with (seed=1) = {}").format(second_session.run(uniform_with_seed))
    print ("uniform without seed = {}").format(second_session.run(uniform_without_seed))
    print ("uniform without seed = {}").format(second_session.run(uniform_without_seed))