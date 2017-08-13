import tensorflow as tf

# a = tf.constant([2, 2], name="a")
# b = tf.constant([3, 6], name="b")
# x = tf.add(a, b, name="add")
#
# with tf.Session() as sess:
#     writer = tf.summary.FileWriter('./graphs', sess.graph)
#     print sess.run(x)
# writer.close()

# W is a random 700 x 100 variable object
# W = tf.Variable(tf.truncated_normal([700, 10]))
# with tf.Session() as sess:
#     sess.run(W.initializer)
#     print W
#
# W = tf.Variable(tf.truncated_normal([700, 10]))
# with tf.Session() as sess:
#     sess.run(W.initializer)
#     print type(W.eval())
#     print W.eval().shape

# create a placeholder of type float 32-bit, shape is a vector of 3 elements
a = tf.placeholder(tf.float32, shape=[3])
# create a constant of type float 32-bit, shape is a vector of 3 elements
b = tf.constant([5, 5, 5], tf.float32)
# use the placeholder as you would a constant or a variable
c = a + b  # Short for tf.add(a, b)
with tf.Session() as sess:
    # feed [1, 2, 3] to placeholder a via the dict {a: [1, 2, 3]}
    # fetch value of c
    writer = tf.summary.FileWriter('./my_graph', sess.graph)
    print(sess.run(c, {a: [1, 2, 3]}))
