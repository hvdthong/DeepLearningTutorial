from tensorflow.contrib import learn
import numpy as np
import tensorflow as tf


def pad_sentences(sentences, seq_len, padding_word="<PAD/>"):
    padded_sentences = []
    for sent in sentences:
        if len(sent.split()) < seq_len:
            num_padding = seq_len - len(sent.split())
            new_sent = sent + " " + " ".join([padding_word] * num_padding)
        else:
            new_sent = sent
        padded_sentences.append(new_sent)
    return padded_sentences


def pad_docs(docs, doc_len, seq_len, padding_word="<PAD/>"):
    new_docs = []
    for doc in docs:
        if len(doc) < doc_len:
            num_padding_doc = doc_len - len(doc)
            for i in xrange(num_padding_doc):
                doc.append(" ".join([padding_word] * seq_len))
        new_docs.append(doc)
    return new_docs


def flat_docs(docs):
    sents = [s for d in docs for s in d]
    return sents

# Start interactive session
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 15])
x_ = tf.reshape(x, [-1, 5])
W_code = tf.Variable(tf.random_uniform([12, 4], -1.0, 1.0), name="W_code")
embedded_chars_code_left = tf.nn.embedding_lookup(W_code, x_[0])
embedded_chars_expanded_code_left = tf.expand_dims(embedded_chars_code_left, -1)


if __name__ == "__main__":
    sents = ["i love deep learning", "it is a great course"]
    sents_1 = ["i love book", "book is great", "reading"]
    docs = [sents, sents_1]
    max_sents = max([len(sent.split()) for sent in sents])
    max_docs = max([len(doc) for doc in docs])
    # pad_sents = pad_sentences(sents, max_sents, padding_word="<PAD/>")
    # pad_docs = pad_docs(docs, doc_len=max_docs, seq_len=max_sents, padding_word="<PAD/>")
    pad_sents = [pad_sentences(doc, max_sents, padding_word="<PAD/>") for doc in docs]
    print pad_sents
    pad_sents = pad_docs(docs=pad_sents, doc_len=max_docs, seq_len=max_sents, padding_word="<PAD/>")
    print pad_sents
    print max_sents, max_docs
    # print len(pad_sents)
    new_sents = flat_docs(pad_sents)
    print new_sents
    print len(new_sents)
    text_vocab_processor = learn.preprocessing.VocabularyProcessor(max_sents)
    x_text = np.array(list(text_vocab_processor.fit_transform(new_sents)))
    print len(text_vocab_processor.vocabulary_)
    exit()
    print x_text.shape
    print x_text
    a = x_text[:(x_text.shape[0] / 2)].reshape(-1, 15)
    print a.reshape(-1, 15)
    print a.reshape(3, 5)
    sess.run(tf.global_variables_initializer())
    train_accuracy = embedded_chars_expanded_code_left.eval(feed_dict={x: [a]})
    # print new_docs
    # print len(new_docs)