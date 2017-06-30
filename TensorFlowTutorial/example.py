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

x = [[1, 0, 0], [1, 0, 0], [0, 0, 1]]
x = np.array(x)
print type(x)
print x.reshape([-1, 9])
print x
exit()
print x[-1]
y = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
correct = np.equal(np.argmax(x, 1), np.argmax(y, 1))
# print correct
accuracy = np.mean(correct)
print correct, accuracy

print np.mean([False, True])

x = np.ones((1, 2, 3))
x_T = np.transpose(x, (1, 0, 2))
print x.shape, x_T.shape
print x[0]
print x_T[0], x_T[1]

x_R = np.reshape(x_T, [-1, 3])
print x_R

# x_S = np.split(0, 3, x_R)
# print x_S
