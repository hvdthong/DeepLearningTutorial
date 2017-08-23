import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32, shape=[None])  # 1-D tensor
i = tf.placeholder(tf.int32, shape=[1])
y = tf.slice(x, i, [1])

#initialize
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

#run
result = sess.run(y, feed_dict={x: [1, 2, 3, 4, 5], i: [0]})
print(result)