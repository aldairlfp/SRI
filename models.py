from collections import Counter
import numpy as np
from collections import defaultdict
import math

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


class Probabilistic(object):
    def __init__(self, docs):
        super().__init__()

        self.documents = docs
        self.document_vectors: list[list[str]] = []
        self.last_ranking: list[tuple[float, int]] = []
        self.query_document_relevance: dict[tuple[str, int], dict[str, float]] = {}

        self.query_document_not_relevance: dict[tuple[str, int], dict[str, float]] = {}
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

    def generate_query_vector(self, query, lang = 'english'):
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

    def get_ranking(self, query, top, lang = 'english'):
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
        for term in query_vector:
            ocurrences = 0
            
            for doc in self.document_vectors:
                if term in doc:
                    ocurrences += 1
                    
            for doc in doc_rank: 
                dr, dnr = self.get_relevance(doc[1], term)
                total = 0
                if term in doc_rank[0].norm_title + doc_rank[0].norm_corpus:
                    total += 1
            
                dr = (dr*5 + total)/(total + 5)
                self.query_document_relevance[(term, doc[1])][term] = dr
                
                dnr = (ocurrences - total)/(len(self.document_vectors) - len(doc_rank))
                self.query_document_not_relevance[(term, doc[1])][term] = dnr
                
    def save_feedback(self):
        file = open("feedback.txt", "w")
        
        for key, value in self.query_document_relevance.items():
            for var in key:
                file.write(str(var) + " ")
            file.write(str(value) + "\n")
        
        for key, value in self.query_document_not_relevance.items():
            for var in key:
                file.write(str(var) + " ")
            file.write(str(value) + "\n")
            
    def load_feedback(self):
        try:
            file = open("feedback.txt", "r")
            for line in file:
                line = line.split()
                self.query_document_relevance[(line[0], int(line[1]))] = {line[0]: float(line[2])}
                self.query_document_not_relevance[(line[0], int(line[1]))] = {line[0]: float(line[2])}
        except:
            pass
                
        
            
        