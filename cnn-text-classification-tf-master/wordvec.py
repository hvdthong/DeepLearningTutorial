import tensorflow as tf
import data_helpers
from tensorflow.contrib import learn
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial


def most_similar_words(vocabulary, wordvec, target, topN):
    if vocabulary.index(target) >= 0:
        target_index = vocabulary.index(target)
        target_wordvec = wordvec[target_index]
        similar_score = [(1 - spatial.distance.cosine(target_wordvec, vec)) for vec in wordvec]
        topN_similar_index = sorted(range(len(similar_score)), key=lambda i: similar_score[i], reverse=True)[:topN + 1]
        topN_word = [vocabulary[i] for i in topN_similar_index]
        topN_score = [similar_score[i] for i in topN_similar_index]
        print topN_word
        print topN_score
        return topN_word, topN_score
    else:
        print 'Word is not appear in dictionary'
        exit()


path = './runs/1500133618/'
tf.flags.DEFINE_string("checkpoint_dir", path + "checkpoints/", "Checkpoint directory from training run")
tf.flags.DEFINE_string("vocabulary", path + "vocab/", "Vocabulary dictionary")
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos",
                       "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg",
                       "Data source for the negative data.")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Load data
print("Loading data...")
x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))
vocab_dict = vocab_processor.vocabulary_._mapping
sorted_vocab = sorted(vocab_dict.items(), key = lambda x : x[1])
vocabulary = list(list(zip(*sorted_vocab))[0])

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        word2vec = graph.get_operation_by_name("embedding/word2vec").outputs[0]
        word2vec = word2vec.eval()

print word2vec.shape, len(vocabulary)
most_similar_words(vocabulary=vocabulary, wordvec=word2vec, target='girl', topN=10)

# with open('vocabulary.txt') as f:
#     vocab_select = [word.strip() for word in f]
#
# vocab_index = [vocab_dict.index(w) for w in vocab_select]
# word2vec_select = np.array([word2vec[i, :] for i in vocab_index])
# print len(vocab_select), word2vec_select.shape
#
# tsne = TSNE(n_components=2)
# np.set_printoptions(suppress=True)
# Y = tsne.fit_transform(word2vec_select)
#
# plt.scatter(Y[:, 0], Y[:, 1])
# for label, x, y in zip(vocab_select, Y[:, 0], Y[:, 1]):
#     plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
# plt.show()
