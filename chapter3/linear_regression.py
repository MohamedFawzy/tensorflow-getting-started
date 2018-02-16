import  os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# build data model
number_of_points = 200

x_points = []
y_points = []

a = 0.22
b = 0.78
# fill x , y list
for i in range(number_of_points):
    x = np.random.normal(0.0, 0.5)
    y = a*x + b + np.random.normal(0.0, 0.1)
    x_points.append([x])
    y_points.append([y])

plt.plot(x_points, y_points, 'o', label="Input Data")
plt.legend()
plt.show()
# variable A with shape 1 and random values between -1,1
A = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
B = tf.Variable(tf.zeros([1]))
y = A * x_points + B

# cost function
cost_function = tf.reduce_mean(tf.square(y - y_points))
# gradient decent function move with 0.5
optimizer = tf.train.GradientDescentOptimizer(0.5)
# train data model
train = optimizer.minimize(cost_function)
# build model
model = tf.global_variables_initializer()
# run session and build variables with 20 steps every step move with 0.5
with tf.Session() as session:
    session.run(model)
    for step in range(0, 21):
        session.run(train)
        if (step % 5) == 0:
            plt.plot(x_points, y_points, 'o', label='step = {}'.format(step))
            plt.plot(x_points, session.run(A) * x_points + session.run(B))
            plt.legend()
            plt.show()
