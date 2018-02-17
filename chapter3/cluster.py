import  os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# k-means algorithm

# number of vectors we want to cluster
num_vectors = 1000
# number of clusters
num_clusters = 4
# number of computational steps of the k-means algorithm
num_steps = 100

# initial input data structures
x_values = []
y_values = []
vector_values = []

for i in range(num_vectors):
    if np.random.random() > 0.5:
        x_values.append(np.random.normal(0.4, 0.7))
        y_values.append(np.random.normal(0.2, 0.8))
    else:
        x_values.append(np.random.normal(0.6, 0.4))
        y_values.append(np.random.normal(0.8, 0.5))

# obtain complete list of the vector list
vector_values = zip(x_values, y_values)
# convert to tensorflow constant
vectors = tf.constant(vector_values)
# plot data

plt.plot(x_values, y_values, 'o', label="Input Data")
plt.legend()
plt.show()
print "vector_values ", vector_values[0]
n_samples = tf.shape(vector_values)[0]
random_indices = tf.random_shuffle(tf.range(0, n_samples))

begin = [0,]
size = [num_clusters, ]
size[0] = num_clusters
print "begin ", begin
print "size ", size
centroid_indices = tf.slice(random_indices, begin, size)
centroids = tf.Variable(tf.gather(vector_values, centroid_indices))
# cost function and optimization
expanded_vectors  = tf.expand_dims(vectors, 0)
expanded_centroids = tf.expand_dims(centroids, 1)

vectors_subtration = tf.subtract(expanded_vectors, expanded_centroids)

euclidean_distances = tf.reduce_sum(tf.square(vectors_subtration), 2)

assignments = tf.to_int32(tf.argmin(euclidean_distances, 0))

partitions = tf.dynamic_partition(vectors, assignments, num_clusters)
for partition in partitions:
    print partition

update_centroids = tf.concat(0,[tf.expand_dims(tf.reduce_mean(partition, 0), 0)for partition in partitions])

init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

for step in xrange(num_steps):
    _, centroid_values, assignment_values = sess.run([update_centroids, centroids, assignments])


def display_partition(x_values, y_values, assignment_values):
    labels = []
    colors = ["red", "blue", "green", "yellow"]
    for i in xrange(len(assignment_values)):
        labels.append(colors[(assignment_values[i])])
    color = labels
    df = pd.DataFrame(dict(x=x_values, y=y_values, color=labels))
    fig, ax = plt.subplots()
    ax.scatter(df['x'], df['y'], c=df['color'])
    plt.show()
