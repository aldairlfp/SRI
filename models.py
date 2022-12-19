from collections import Counter
import numpy as np
from collections import defaultdict
import math
import os

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


class Probabilistic(object):
    def __init__(self, docs):
        super().__init__()

        self.documents = docs
        self.document_vectors: list[list[str]] = []
        self.last_ranking: list[tuple[float, int]] = []
        self.query_document_relevance: dict[tuple[str, int], dict[str, float]] = {}
        self.query_document_not_relevance: dict[tuple[str, int], dict[str, float]] = {}

        self.load_feedback()
        """
            This contains the relevance of a query in a document
        """

    def generate_document_vectors(self):
        """Generate the document vectors for the model

        :param self: The model
        :type self: Probabilistic

        :return: None
        :rtype: None
        """
        self.document_vectors = []

        for doc_index, doc in enumerate(self.documents):
            self.document_vectors.append([])
            for term in (doc.norm_title + doc.norm_corpus):
                self.document_vectors[doc_index].append(term)

    def get_term_frequency(self, term: str) -> float:
        n_i = 0
        for doc in self.document_vectors:
            if term in doc:
                n_i += 1
        return n_i / len(self.documents)

    def generate_query_vector(self, query, lang='english'):
        return default_processor(query, lang)

    def get_relevance(self, document_id, term):
        """Get the relevance of a term in a document for a query

        :param self: The model
        :type self: Probabilistic

        :param query: The query
        :type query: str

        :param document_id: The document id
        :type document_id: int

        :param term: The term
        :type term: str

        :return: The relevance of the term in the document for the query
        :rtype: float
        """
        if (term, document_id) in self.query_document_relevance:
            dr = self.query_document_relevance[(term, document_id)][term]
            dnr = self.query_document_not_relevance[(term, document_id)][term]
            return dr, dnr

        dr = 0.5
        dnr = self.get_term_frequency(term)

        self.query_document_relevance[(term, document_id)] = {term: dr}
        self.query_document_not_relevance[(term, document_id)] = {term: dnr}

        return dr, dnr

    def similarity(self, query, document, document_id) -> float:
        """Calculate the similarity between a query and a document

        :param self: The model
        :type self: Probabilistic

        :param raw_query: The raw query
        :type raw_query: str

        :param query: The query vector
        :type query: list[str]

        :param document: The document vector
        :type document: list[str]

        :param document_id: The document id
        :type document_id: int

        :return: The similarity between the query and the document
        :rtype: float
        """
        similarity = 0

        for common_term in query:
            if common_term in document:
                p_i, r_i = self.get_relevance(document_id, common_term)
                similarity += math.log((p_i * (1 - r_i)) / (r_i * (1 - p_i)))

        return 0 if similarity == 0 else math.log(similarity)

    def get_ranking(self, query, top, lang='english'):
        """Get the ranking of a query

        :param self: The model
        :type self: Probabilistic

        :param query: The user query
        :type query: str

        :param top: The number of documents to be returned
        :type top: int

        :param lang: The language of the query
        :type lang: str

        :return: The list of documents ranked
        :rtype: list[Document]
        """

        query_vector = self.generate_query_vector(query, lang)
        self.generate_document_vectors()
        doc_rank = []

        for index, _ in enumerate(self.documents):
            doc_vector = self.document_vectors[index]
            sim = self.similarity(query_vector, doc_vector, index)

            if sim > 0.5:
                doc_rank.append((sim, index))
        self.last_ranking = sorted(
            doc_rank, key=lambda rank_index: rank_index[0], reverse=True)

        self.feedback(query, doc_rank)

        return [self.documents[x[1]] for x in self.last_ranking[:top]]

    def feedback(self, query, doc_rank):
        """Give feedback to the model

        :param self: The model
        :type self: Probabilistic

        :param query: The query
        :type query: str

        :param doc_rank: The final rank for the query
        :type doc_rank: list[Document]

        :return: None
        :rtype: None
        """
        query_vector = self.generate_query_vector(query)
        self.load_feedback()

        for term in query_vector:
            ocurrences = 0

            for doc in self.document_vectors:
                if term in doc:
                    ocurrences += 1

            for doc in [(self.document_vectors[x[1]], x[1]) for x in doc_rank]:
                dr, dnr = self.get_relevance(doc[1], term)
                total = 0

                if term in doc[0]:
                    total += 1

                dr = (dr * 10 + total) / (total + 10)
                self.query_document_relevance[(term, doc[1])][term] = dr

                dnr = (ocurrences - total) / (len(self.document_vectors) - len(doc_rank))
                self.query_document_not_relevance[(term, doc[1])][term] = dnr

        self.save_feedback()

    def save_feedback(self):
        file = open("feedback.txt", "w")

        file.write("query document relevance\n")
        for key, value in self.query_document_relevance.items():
            for var in key:
                file.write(str(var) + " ")
            file.write(str(value) + "\n")

        file.write("query document not relevance\n")
        for key, value in self.query_document_not_relevance.items():
            for var in key:
                file.write(str(var) + " ")
            file.write(str(value) + "\n")

    def load_feedback(self):
        try:
            file = open("feedback.txt", "r")
            lines = file.readlines()
            index = 0
            for line in lines:
                index += 1
                if line == "query document relevance\n":
                    continue
                if 'not relevance' in line:
                    break
                line = line.split()
                self.query_document_relevance[(line[0], int(line[1]))] = {line[0]: float(line[3][:-1])}

            for line in lines[index:]:
                line = line.split()
                self.query_document_not_relevance[(line[0], int(line[1]))] = {line[0]: float(line[3][:-1])}
        except FileNotFoundError:
            pass


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
