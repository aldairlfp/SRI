from collections import OrderedDict, Counter

from nltk.corpus import stopwords
from nltk import word_tokenize
import string
import numpy as np

from data_collections import default_processor


class VectorSpace(object):
    def __init__(self, docs):
        M = len(docs)  # number of files in dataset
        self.docs = docs
        self._tf_dict = self._termFrequencyInDoc()  # returns term frequency
        self._df_dict = self._wordDocFre()  # returns document frequencies
        self._idf_dict = self._inverseDocFre(M)  # returns idf scores
        self._tf_idf = self._tfidf(docs)  # returns tf-idf scores
        self.a = 0.5

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
        tf_idf_scr = [{} for doc in self.docs]
        for i, doc in enumerate(self.docs):
            for word in doc.norm_corpus:
                try:
                    tf = self._tf_dict[i][word]
                    idf = self._idf_dict[word]
                    tf_idf_scr[i][word] = tf * idf
                except KeyError:
                    pass
        return tf_idf_scr

    def ranking(self, query, top):
        query = default_processor(query, 'english')
        # create counter for query
        query_counter = Counter(query)
        # get from counter max frequency
        max_freq = query_counter.most_common(1)[0][1]
        query = [word for word in query if word in self._idf_dict]
        query_tfidf = {}
        for word in query:
            if word in query_tfidf:
                query_tfidf[word] += 1
            else:
                query_tfidf[word] = 1
        for word in query_tfidf:
            query_tfidf[word] = (self.a + (1 - self.a) * query_tfidf[word] / max_freq) * self._idf_dict[word]

        scores = [0 for doc in self.docs]
        for i, doc in enumerate(self.docs):
            for word in query_tfidf:
                if word in self._tf_idf[i]:
                    scores[i] += self._tf_idf[i][word] * query_tfidf[word] / np.sqrt(
                        sum([self._tf_idf[i][word] ** 2 for word in self._tf_idf[i]]))
        scores = np.array(scores)
        scores = scores.argsort()[-top:][::-1]
        return [self.docs[i] for i in scores]
