from nltk.tokenize import RegexpTokenizer
import pandas as pd

tokenizer0 = RegexpTokenizer(r"(?u)\b\w\w+\b")
SEED = 2016
vocabulary_size = 50000


def read_treebank(nrows=100):
    '''
    Read sentences from treebank datasetSentences.txt file and tokenize them.
    Return a list of token lists, each of which is from the same sentence.
    '''
    df = pd.read_csv('data/datasetSentences.txt',
                     sep='\t',
                     nrows=nrows)
    # return df['sentence'].map(word_tokenize).tolist()
    return df['sentence'].map(tokenizer0.tokenize).tolist()


docs = read_treebank()


def concat_lists(a, b):
    '''Concat two lists
    '''
    # a.extend(b)
    return a + b
