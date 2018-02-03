import os
import matplotlib.image as mp_image
import matplotlib.pyplot as plt
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


file_name = "test.png"
input_image = mp_image.imread(file_name)

my_image = tf.placeholder("uint8", [None, None, 4])

slice = tf.slice(my_image, [10, 0, 0], [16, -1, -1])

with tf.Session() as session:
    result = session.run(slice, feed_dict={my_image: input_image})
    print (result.shape)

plt.imshow(result)
plt.show()
