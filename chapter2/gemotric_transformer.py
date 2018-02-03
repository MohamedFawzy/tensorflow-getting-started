import tensorflow as tf
import os
import matplotlib.image as mp_image
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

file_name = "test.png"
input_image = mp_image.imread(file_name)

my_image = tf.placeholder("uint8", [None, None, 4])


x = tf.Variable(input_image, name="x")

model = tf.global_variables_initializer()

with tf.Session() as session:
    x = tf.transpose(x, perm=[1, 0, 2])
    session.run(model)
    result = session.run(x)

plt.imshow(result)
plt.show()