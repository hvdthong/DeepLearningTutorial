import numpy as np
import tensorflow as tf

h = [2, 1, 0]
x = [3, 4, 5]

y = np.convolve(x, h, 'full')
print y

x = [6, 2]
h = [1, 2, 5, 4]

y = np.convolve(x, h, "full")  # now, because of the zero padding, the final dimension of the array is bigger
print y

from scipy import signal as sg

I = [[255, 7, 3],
     [212, 240, 4],
     [218, 216, 230], ]

g = [[-1, 1]]

print ('Without zero padding \n')
print ('{0} \n'.format(sg.convolve(I, g, 'valid')))
# The 'valid' argument states that the output consists only of those elements
# that do not rely on the zero-padding.

print ('With zero padding \n')
print sg.convolve(I, g)

# Building graph

input = tf.Variable(tf.random_normal([1, 3, 3, 1]))
filter = tf.Variable(tf.random_normal([2, 2, 1, 1]))
op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
op2 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')
texting = tf.Variable(tf.truncated_normal([3, 3, 1, 3], stddev=0.1))

# Initialization and session
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    # print("Input \n")
    # print('{0} \n'.format(input.eval()))
    # # print input.eval().shape
    # print("Filter/Kernel \n")
    # print('{0} \n'.format(filter.eval()))
    # # print filter.eval().shape
    # print("Result/Feature Map with valid positions \n")
    # result = sess.run(op)
    # print(result)
    # print result.shape
    # print('\n')
    # print("Result/Feature Map with padding \n")
    # result2 = sess.run(op2)
    # print(result2)
    # print result2.shape

    result3 = sess.run(texting)
    print result3
    print result3.shape
