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
embedding_dim, filter_sizes, num_filters, dropout_keep_prob = 128, [3, 4, 5], 128, 0.5
