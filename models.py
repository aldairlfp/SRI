from collections import Counter
import numpy as np

from parse import BooleanExpression
from data_collections import default_processor
import utils


class Query(object):
    def __init__(self, text: str, cr: list, cnr: list):
        self.text: str = text
        self.cr: list = cr
        self.cnr: list = cnr

    def set_relevance(self, d):
        if d not in self.cr:
            self.cr.append(d)
        if d in self.cnr:
            self.cnr.remove(d)

    def set_non_relevance(self, d):
        if d not in self.cnr:
            self.cnr.append(d)
        if d in self.cr:
            self.cr.remove(d)


class Model(object):
    def __init__(self):
        self.querys = {}

    def fix_relevance(self, query):
        for doc in self.querys[query].cr:
            if not doc.is_relevant:
                self.querys[query].cr.remove(doc)
                self.querys[query].cnr.append(doc)
        for doc in self.querys[query].cnr:
            if doc.is_relevant:
                self.querys[query].cnr.remove(doc)
                self.querys[query].cr.append(doc)

    def set_relevance(self, query, d):
        if query not in self.querys:
            self.querys[query] = Query(query, [d], [])
        elif d not in self.querys[query].cr:
            self.querys[query].cr.append(d)
        if d in self.querys[query].cnr:
            self.querys[query].cnr.remove(d)

    def set_non_relevance(self, query, d):
        if query not in self.querys:
            self.querys[query] = Query(query, [], [d])
        elif d not in self.querys[query].cnr:
            self.querys[query].cnr.append(d)
        if d in self.querys[query].cr:
            self.querys[query].cr.remove(d)

    def calculate_weights_query(self, query):
        raise NotImplementedError

    def create_query(self, text):
        raise NotImplementedError


class VectorSpace(Model):
    def __init__(self, docs, corpus_type):
        super().__init__()
        M = len(docs)  # number of files in dataset
        self.docs = docs
        self._corpus_type = corpus_type
        self._tf_dict = utils.termFrequencyInDoc(self.docs)  # returns term frequency
        self._df_dict = utils.wordDocFre(self.docs)  # returns document frequencies
        self._idf_dict = utils.inverseDocFre(self._df_dict, M)  # returns idf scores
        self._tf_idf = utils.tfidf(self.docs, self._tf_dict, self._idf_dict)  # returns tf-idf scores

    def calculate_weights_query(self, query, a=0.5):
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
            query_tfidf[word] = (a + (1 - a) * query_tfidf[word] / max_freq) * self._idf_dict[word]

        return query_tfidf

    def sim(self, query_tfidf):
        scores = [0 for doc in self.docs]
        for i, doc in enumerate(self.docs):
            for word in query_tfidf:
                if word in self._tf_idf[i]:
                    scores[i] += self._tf_idf[i][word] * query_tfidf[word] / np.sqrt(
                        sum([self._tf_idf[i][word] ** 2 for word in self._tf_idf[i]]))
        return scores

    def ranking(self, query):
        if query not in self.querys:
            self.create_query(query)
            query_tfidf = self.calculate_weights_query(query)
        else:
            self.fix_relevance(query)
            query_tfidf = self.rocchio(query, self.querys[query].cr, self.querys[query].cnr)

        scores = self.sim(query_tfidf)

        scores_index = np.array(scores)
        scores_index = scores_index.argsort()[:][::-1]
        return [self.docs[i] for i in scores_index if scores[i] > 0]

    def create_query(self, text):
        self.querys[text] = Query(text, [], [])

    def rocchio(self, query, cr, cnr, alpha=1, beta=0.75, gamma=0.15):
        """
        Rocchio algorithm
        :param query: query
        :param cr: relevant documents
        :param cnr: non-relevant documents
        :param alpha: weight of query
        :param beta: weight of relevant documents
        :param gamma: weight of non-relevant documents
        :return: new query
        """
        qm = self.calculate_weights_query(query)

        for word in qm:
            qm[word] *= alpha

        for doc in cr:
            doc_tfidf = self._tf_idf[doc.id]
            for word in doc_tfidf:
                if word in qm:
                    qm[word] += beta * doc_tfidf[word] / len(cr)
                else:
                    qm[word] = doc_tfidf[word]

        for doc in cnr:
            doc_tfidf = self._tf_idf[doc.id]
            for word in doc_tfidf:
                if word in qm:
                    qm[word] -= gamma * doc_tfidf[word] / len(cnr)
                else:
                    qm[word] = doc_tfidf[word]

        return qm

    def __str__(self):
        return "vector_space" + '_' + self._corpus_type


class BooleanExtended(Model):
    def __init__(self, docs, corpus_type):
        super().__init__()
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

    def ranking(self, r_query):
        """ 
        Method to rank the documents based on the query

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
        scores = scores.argsort()[:][::-1]
        return [self.docs[i] for i in scores if rank[i] > 0]

    def dnf_method(self, query, qcount=2):
        """
        Method to evaluate the query using the DNF method

        :param qcount: adjusts the occurrence characteristics of the query terms
        :param self: instance of the model
        :type self: Extended

        :param query: query
        :type query: str

        :rtype: list
        :return: list of ranked documents
        """

        # tokenize the query
        query = query.split()

        # parse the query
        expr = BooleanExpression('or')
        expr.parse(query)

        # R = self.querys[query].cr
        # for c in expr.children:
        #     if c.type == 'and':
        #
        # TODO: implement the DNF method


    def __str__(self):
        return "extended_boolean" + "_" + self._corpus_type
