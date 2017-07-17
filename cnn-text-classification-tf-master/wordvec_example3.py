import data_helpers
import tensorflow as tf
from gensim.models import Word2Vec


tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos",
                       "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg",
                       "Data source for the negative data.")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
x_text_new = []
for line in x_text:
    x_text_new.append(line.split(' '))

model = Word2Vec(x_text_new, size=100, window=5, min_count=0, workers=4, iter=30)
# model.wv.save_word2vec_format("gensim_word2vec.txt",fvocab=None,binary=False)
print model.most_similar('girl')