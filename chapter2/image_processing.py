import matplotlib.image as mp_image
import matplotlib.pyplot as plt
import tensorflow as tf

file_name = "test.png"
input_image = mp_image.imread(file_name)

print 'input dim = {} '.format(input_image.ndim)
print 'input shape= {}'.format(input_image.shape)

plt.imshow(input_image)
plt.show()

my_image = tf.placeholder("uint8", [None, None, 3])

