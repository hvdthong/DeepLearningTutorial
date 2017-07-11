import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn

pos_path = './data/rt-polaritydata/rt-polarity.pos'
neg_path = './data/rt-polaritydata/rt-polarity.neg'
embedding_dim, filter_sizes, num_filters, dropout_keep_prob, l2_reg_lambda = 128, [3, 4, 5], 128, 0.5, 0
batch_size, num_epochs = 64, 200

# Load data
print("Loading data...")
x_text, y = data_helpers.load_data_and_labels(pos_path, neg_path)
print len(x_text)
print len(y)

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]
print x_shuffled.shape, y_shuffled.shape
# print x_shuffled[0], y_shuffled[0]

# Training
# ==================================================
with tf.Session() as session:
    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)
    sess = tf.Session(config=session_conf)
    cnn = TextCNN(
        sequence_length=x_shuffled.shape[1],
        num_classes=y_shuffled.shape[1],
        vocab_size=len(vocab_processor.vocabulary_),
        embedding_size=embedding_dim,
        filter_sizes=filter_sizes,
        num_filters=num_filters,
        l2_reg_lambda=l2_reg_lambda)

    # Define Training procedure
    # global_step = tf.Variable(0, name="global_step", trainable=False)
    # optimizer = tf.train.AdamOptimizer(1e-3)
    loss, accuracy = cnn.loss, cnn.accuracy
    optimizer = tf.train.AdamOptimizer().minimize(cnn.loss)

    # grads_and_vars = optimizer.compute_gradients(cnn.loss)
    # train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)


    def train_step(batch, x_batch, y_batch):
        """
        A single training step
        """
        feed_dict = {
            cnn.input_x: x_batch,
            cnn.input_y: y_batch,
            cnn.dropout_keep_prob: dropout_keep_prob
        }
        loss_, accuracy_ = sess.run(
            [loss, accuracy],
            feed_dict=feed_dict)

        print("step {}, loss {:g}, acc {:g}".format(batch, loss_, accuracy_))

    # Generate batches
    batches = data_helpers.batch_iter(
        list(zip(x_shuffled, y_shuffled)), batch_size, num_epochs)

    # Training loop. For each batch...
    for batch in batches:
        x_batch, y_batch = zip(*batch)
        train_step(batch, x_batch, y_batch)
        # current_step = tf.train.global_step(sess, global_step)
#