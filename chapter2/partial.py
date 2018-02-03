import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import  os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


def make_kernel(a):
  """Transform a 2D array into a convolution kernel"""
  a = np.asarray(a)
  a = a.reshape(list(a.shape) + [1,1])
  return tf.constant(a, dtype=1)

def simple_conv(x, k):
  """A simplified 2D convolution operation"""
  x = tf.expand_dims(tf.expand_dims(x, 0), -1)
  y = tf.nn.depthwise_conv2d(x, k, [1, 1, 1, 1], padding='SAME')
  return y[0, :, :, 0]

def laplace(x):
  """Compute the 2D laplacian of an array"""
  laplace_k = make_kernel([[0.5, 1.0, 0.5],
                           [1.0, -6., 1.0],
                           [0.5, 1.0, 0.5]])
  return simple_conv(x, laplace_k)



sess = tf.InteractiveSession()

N = 500

# Initial Conditions -- some rain drops hit a pond

# Set everything to zero
u_init = np.zeros([N, N], dtype=np.float32)
ut_init = np.zeros([N, N], dtype=np.float32)

# Some rain drops hit a pond at random points
for n in range(100):
  a,b = np.random.randint(0, N, 2)
  u_init[a,b] = np.random.uniform()

plt.imshow(u_init)
plt.show()

# model building

eps = tf.placeholder(tf.float32, shape=())
damping = tf.placeholder(tf.float32, shape=())
U  = tf.Variable(u_init)
Ut = tf.Variable(ut_init)

# PDE model
U_ = U + eps * Ut
Ut_ = Ut + eps * (laplace(U) - damping * Ut)

step = tf.group(U.assign(U_), Ut.assign(Ut_))

tf.global_variables_initializer().run()

for i in range(1000):
    step.run({eps: 0.03, damping: 0.04})
    if i % 50 == 0:
        plt.imshow(U.eval())
        plt.show()
