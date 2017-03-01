import tensorflow as tf
import numpy as np

# x1 = tf.constant(5)
# x2 = tf.constant(6)
#
# result = tf.mul(x1, x2)
# print result
#
# # sess = tf.Session()
# # print (sess.run(result))
# # sess.close()
#
# # do not need to remember close session
# with tf.Session() as sess:
#     output = sess.run(result)
#     print output
#
# print output
#
# x = [4, 5, 6, 2, 1]
# print x[:-2]
# print x[-2:]
#
# print x[:(len(x) - 2)]
# print x[(len(x) - 2):]
#
# x = np.ones((1, 2, 3))
# xT = np.transpose(x, (1, 0, 2))
# print x, x[0]
# print xT, x[0]
# print x.shape, xT.shape
# print np.ones((1, 2))

x = [[1, 0, 0], [1, 0 , 0], [0, 0, 1]]
y = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
correct = np.equal(np.argmax(x, 1), np.argmax(y, 1))
# print correct
accuracy = np.mean(correct)
print correct, accuracy

print np.mean([False, True])

