from gensim.models.word2vec import Word2Vec
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_20newsgroups
import re
import matplotlib.pyplot as plt

# download example data ( may take a while)
train = fetch_20newsgroups()


def clean(text):
    """Remove posting header, split by sentences and words, keep only letters"""
    lines = re.split('[?!.:]\s', re.sub('^.*Lines: \d+', '', re.sub('\n', ' ', text)))
    return [re.sub('[^a-zA-Z]', ' ', line).lower().split() for line in lines]

sentences = [line for text in train.data for line in clean(text)]

model = Word2Vec(sentences, workers=4, size=100, min_count=50, window=10, sample=1e-3)

print (model.most_similar('memory'))

X = model[model.wv.vocab]
# print X.shape
# print X[0]
# exit()
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
plt.show()

