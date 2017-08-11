import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell

mnist = input_data.read_data_sets("/train/data/", one_hot=True)

n_classes = 10
hm_epochs = 3
batch_size = 128

# size of images: 28x28
chunk_size, n_chunks, rnn_size = 28, 28, 128

#  height x width
x = tf.placeholder('float', [None, n_chunks, chunk_size])  # 28x28
y = tf.placeholder('float')


def neural_network_model(x):
    layer = {'weights': tf.Variable(tf.random_normal([rnn_size, n_classes])),
             'biases': tf.Variable(tf.random_normal([n_classes]))}
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(0, n_chunks, x)

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size, state_is_tuple=True)
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.add(tf.matmul(outputs[-1], layer['weights']), layer['biases'])  # get the latest output
    return output


def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))

    # learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    x_T = tf.transpose(x, [1, 0, 2])
    # feedforward and backward

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                # print epoch_x.shape
                epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size))
                print epoch_x.shape

                x_T = x_T.eval({x:epoch_x})
                print x_T.shape
                break

                # epoch_x_T = epoch_x.transpose((1, 0, 2))
                # print epoch_x_T.shape
                # epoch_x_R = epoch_x_T.reshape((-1, chunk_size))
                # print epoch_x_R.shape
                # epoch_x_S = epoch_x_R.split(epoch_x_R, n_chunks)
                # print epoch_x_S.shape

                # print epoch_x_T.shape, epoch_x_R.shape, epoch_x_S.shape

                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})

                epoch_loss += c

            print 'Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print 'Accuracy:', accuracy.eval(
            {x: mnist.test.images.reshape((-1, n_chunks, chunk_size)), y: mnist.test.labels})


train_neural_network(x)

# print mnist.train.num_examples
# print mnist.test.images.shape
# epoch_x, epoch_y = mnist.train.next_batch(batch_size)
# print epoch_x.shape, epoch_y.shape
