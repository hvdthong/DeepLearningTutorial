import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print mnist

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
# Weight tensor
W = tf.Variable(tf.zeros([784, 10], tf.float32))
# Bias tensor
b = tf.Variable(tf.zeros([10], tf.float32))

# run the op initialize_all_variables using an interactive session
sess.run(tf.global_variables_initializer())

# mathematical operation to add weights and biases to the inputs
tf.matmul(x, W) + b
y = tf.nn.softmax(tf.matmul(x, W) + b)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Load 50 training examples for each training iteration
for i in range(10000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    acc = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}) * 100
    if i % 100 == 0:
        print("The final accuracy for the simple ANN model is: %.3f at iteration %i" % (acc, i))
sess.close()  # finish the session
