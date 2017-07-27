import numpy as np
import tensorflow as tf
import random

# matrix = np.random.random([1024, 64])  # 64-dimensional embeddings
# ids = np.array([0, 5, 17, 33])
# print matrix[ids]  # prints a matrix of shape [4, 64]

# t2 = [2, 3, 5]
# t2 = tf.constant(t2)
# W = tf.random_uniform([5, 3], -1.0, 1.0)
# embedded_chars = tf.nn.embedding_lookup(W, 0)
# embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
# init_op = tf.global_variables_initializer()
# with tf.Session() as session:
#     print session.run(t2)
#     print W.eval()
#     print embedded_chars.eval()
#     print embedded_chars_expanded.eval()
#
# m = 4
# x = list(range(0, 3000, 3))
# y = list(range(0, 2000, 2))
# print x
# print y
# print random.sample(list(zip(x, y)), m)
#
# tf1 = tf.random_uniform([1, 384])
# tf2 = tf.random_uniform([384, 384])
# init_op = tf.global_variables_initializer()
# with tf.Session() as session:
#     value = tf.matmul(tf1, tf2)
#     print value.eval().shape

# make a pair
m = 3
x = np.arange(50).T
y = np.arange(10).T
print x
print y
pairs = np.hstack([x[np.random.choice(len(x), m)], y[np.random.choice(len(y), m)]])

print pairs

c = np.array([3., 6, 7])
print(np.mean(c, 0))

Mean = tf.reduce_mean(c)
with tf.Session() as sess:
    result = sess.run(Mean)
    print(result)

# # Creates a graph.
with tf.device('/gpu:0'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
  c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
# Runs the op.
print(sess.run(c))

# # Creates a graph.
# a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
# b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
# c = tf.matmul(a, b)
# # Creates a session with log_device_placement set to True.
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# # Runs the op.
# print(sess.run(c))