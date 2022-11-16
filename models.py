from collections import OrderedDict

from nltk.corpus import stopwords
from nltk import word_tokenize
import string
import numpy as np


class VectorSpace(object):
    def __init__(self, docs):
        M = len(docs)  # number of files in dataset
        self.docs = docs
        self._tf_dict = self._termFrequencyInDoc()  # returns term frequency
        self._df_dict = self._wordDocFre()  # returns document frequencies
        self._idf_dict = self._inverseDocFre(M)  # returns idf scores
        self._tf_idf = self._tfidf(docs)  # returns tf-idf scores

    def _termFrequencyInDoc(self):
        tf_docs = [{} for doc in self.docs]

        for doc in self.docs:
            for i, word in enumerate(doc.norm_corpus):
                if word in tf_docs[i]:
                    tf_docs[i][word] += 1
                else:
                    tf_docs[i][word] = 1

        return tf_docs

    def _wordDocFre(self):
        df = {}
        for doc in self.docs:
            for word in doc.norm_corpus:
                if word in df:
                    df[word] += 1
                else:
                    df[word] = 1
        return df

    def _inverseDocFre(self, length):
        idf = {}
        for word in self._df_dict:
            idf[word] = np.log10(length / self._df_dict[word])
        return idf

    def _tfidf(self, doc):
        tf_idf_scr = {}
        for i, doc in enumerate(self.docs):
            for word in doc.norm_corpus:
                try:
                    tf = self._tf_dict[i][word]
                    idf = self._idf_dict[word]
                    tf_idf_scr[word] = tf * idf
                except KeyError:
                    pass
        return tf_idf_scr

    def make_query(self, query):
        query_vocab = []
        for word in query.split():
            if word not in query_vocab:
                query_vocab.append(word)

        query_wc = {}
        for word in query_vocab:
            query_wc[word] = query.lower().split().count(word)

        relevance_scores = {}
        for i, doc in enumerate(self.docs):
            score = 0
            for word in query_vocab:
                try:
                    score += query_wc[word] * self._tf_idf[i][word]
                except KeyError:
                    pass
            relevance_scores[i] = score
        sorted_value = OrderedDict(sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True))
        top_5 = {k: sorted_value[k] for k in list(sorted_value)[:5]}
        return top_5
