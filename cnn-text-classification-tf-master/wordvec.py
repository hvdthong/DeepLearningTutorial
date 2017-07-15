import tensorflow as tf
import data_helpers
from tensorflow.contrib import learn


path = './runs/1500104294/'
tf.flags.DEFINE_string("checkpoint_dir", path + "checkpoints/", "Checkpoint directory from training run")
tf.flags.DEFINE_string("vocabulary", path + "vocab/", "Vocabulary dictionary")
# tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos",
#                        "Data source for the positive data.")
# tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg",
#                        "Data source for the negative data.")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# # Load data
# print("Loading data...")
# x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
#
# # Build vocabulary
# max_document_length = max([len(x.split(" ")) for x in x_text])
# vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)

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

        # saver = tf.train.import_meta_graph(FLAGS.vocabulary)
        # saver.restore(sess, FLAGS.vocabulary)

        vocab_processor = learn.preprocessing.VocabularyProcessor.restore(FLAGS.vocabulary)
        vocab_dict = vocab_processor.vocabulary_._mapping
        print type(vocab_dict)
        print word2vec.shape

# for word in vocab_dict:
#     print word
