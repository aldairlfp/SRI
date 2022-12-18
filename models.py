from collections import Counter
import numpy as np

from documents import Document
from parse import BooleanExpression
from data_collections import default_processor
import utils


class Model(object):
    def calculate_weights_query(self, query):
        raise NotImplementedError


class VectorSpace(Model):
    def __init__(self, docs, corpus_type):
        M = len(docs)  # number of files in dataset
        self.docs = docs
        self._corpus_type = corpus_type
        self._tf_dict = utils.termFrequencyInDoc(self.docs)  # returns term frequency
        self._df_dict = utils.wordDocFre(self.docs)  # returns document frequencies
        self._idf_dict = utils.inverseDocFre(self._df_dict, M)  # returns idf scores
        self._tf_idf = utils.tfidf(self.docs, self._tf_dict, self._idf_dict)  # returns tf-idf scores
        self.a = 0.5

    def calculate_weights_query(self, query):
        query = default_processor(query, 'english')
        # create counter for query
        query_counter = Counter(query)
        try:
            # get from counter max frequency
            max_freq = query_counter.most_common(1)[0][1]
        except IndexError:
            max_freq = 0
        query = [word for word in query if word in self._idf_dict]
        query_tfidf = {}
        for word in query:
            if word in query_tfidf:
                query_tfidf[word] += 1
            else:
                query_tfidf[word] = 1
        for word in query_tfidf:
            query_tfidf[word] = (self.a + (1 - self.a) * query_tfidf[word] / max_freq) * self._idf_dict[word]

        return query_tfidf

    def ranking(self, query):
        query_tfidf = self.calculate_weights_query(query)

        scores = [0 for doc in self.docs]
        for i, doc in enumerate(self.docs):
            for word in query_tfidf:
                if word in self._tf_idf[i]:
                    scores[i] += self._tf_idf[i][word] * query_tfidf[word] / np.sqrt(
                        sum([self._tf_idf[i][word] ** 2 for word in self._tf_idf[i]]))

        scores_index = np.array(scores)
        scores_index = scores_index.argsort()[:][::-1]
        return [self.docs[i] for i in scores_index if scores[i] > 0]

    def rocchio(self, query, alpha, beta, gamma, cr, cnr):
        qm = self.calculate_weights_query(query)

        for word in qm:
            qm[word] *= alpha

        for doc in cr:
            doc_tfidf = self._tf_idf[doc.id]
            for word in qm:
                if word in doc_tfidf:
                    qm[word] += beta * doc_tfidf[word] / len(cr)
                else:
                    qm[word] = doc_tfidf[word]

        for doc in cnr:
            doc_tfidf = self._tf_idf[doc.id]
            for word in qm:
                if word in doc_tfidf:
                    qm[word] -= gamma * doc_tfidf[word] / len(cnr)
                else:
                    qm[word] = doc_tfidf[word]

        return qm


    def __str__(self):
        return "vector_space" + '_' + self._corpus_type


class BooleanExtended(object):
    def __init__(self, docs, corpus_type):
        self.docs = docs
        self._corpus_type = corpus_type
        self._tf_dict = utils.termFrequencyInDoc(self.docs)  # returns term frequency
        self._df_dict = utils.wordDocFre(self.docs)
        self._idf_dict = utils.inverseDocFre(self._df_dict, len(self.docs))  # returns idf scores
        self._tf_idf = utils.tfidf(self.docs, self._tf_dict, self._idf_dict)  # returns tf-idf scores
        self._norm_frec = utils.normalize_frequency(self.docs, self._tf_idf)
        self._weights = self.weight()

    def weight(self):
        """
        Calculate the weight of each word in each document

        :param self: instance of the model
        :type self: BooleanExtended

        :rtype: list
        :return: weight
        """

        weight = []
        for i, doc in enumerate(self.docs):
            weight.append({})
            for j, word in enumerate(doc.norm_corpus):
                try:
                    weight[i][word] = self._norm_frec[i][j] * self._idf_dict[word] / max(self._idf_dict.values())
                except KeyError:
                    weight[i][word] = 0
        return weight

    def ranking(self, r_query, top):
        """ 
        Method to rank the documents based on the query
        
        :param top:
        :param self: instance of the model
        :type self: Extended
        
        :param r_query: query
        :type r_query: str
        
        :rtype: list
        :return: list of ranked documents
        """

        # tokenize the query
        query = r_query.split()

        # parse the query
        expr = BooleanExpression('or')
        expr.parse(query)

        # evaluate the query
        rank = expr.eval(self)

        scores = np.array(rank)
        scores = scores.argsort()[-top:][::-1]
        return [self.docs[i] for i in scores]

    def __str__(self):
        return "extended_boolean" + "_" + self._corpus_type
