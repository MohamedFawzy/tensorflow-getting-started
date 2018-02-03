import os
import tensorflow as tf
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

tensor_1d = np.array([1, 2, 3, 4, 5, 6, 7])

tf_tensor = tf.convert_to_tensor(tensor_1d, dtype=tf.float64)

with tf.Session() as session:
    print session.run(tf_tensor)
    print session.run(tf_tensor[0])
    print session.run(tf_tensor[1])


tensor_2d = np.array([(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12)])
print tensor_2d
print tensor_2d[1][2]
print tensor_2d[0:2, 0:2]

matrix1 = np.array([(2, 2, 2), (2, 2, 2), (2, 2, 2)], dtype="int32")
matrix2 = np.array([(1, 1, 1), (1, 1, 1), (1, 1, 1)], dtype="int32")

print "matrix1 = "
print matrix1
print "matrix2 = "
print matrix2


matrix1 = tf.constant(matrix1)
matrix2 = tf.constant(matrix2)

matrix_product = tf.matmul(matrix1, matrix2)
matrix_sum = tf.add(matrix1, matrix2)


matrix_3 = np.array([(2, 7, 2), (1, 4, 2), (9, 0, 2)], dtype="float32")

print "matrix 3 = "
print matrix_3

matrix_det = tf.matrix_determinant(matrix_3)

with tf.Session() as sess:
    result1 = sess.run(matrix_product)
    result2 = sess.run(matrix_sum)
    result3 = sess.run(matrix_det)

print "matrix product ="
print result1

print "matrix_sum= "
print result2

print "matrix_determinant result = "
print result3
