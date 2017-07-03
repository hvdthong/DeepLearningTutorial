import numpy as np
import tensorflow as tf


# matrix = np.random.random([1024, 64])  # 64-dimensional embeddings
# ids = np.array([0, 5, 17, 33])
# print matrix[ids]  # prints a matrix of shape [4, 64]

t2 = [2, 3, 5]
t2 = tf.constant(t2)
W = tf.random_uniform([5, 3], -1.0, 1.0)
embedded_chars = tf.nn.embedding_lookup(W, 0)
embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
init_op = tf.global_variables_initializer()
with tf.Session() as session:
    print session.run(t2)
    print W.eval()
    print embedded_chars.eval()
    print embedded_chars_expanded.eval()

