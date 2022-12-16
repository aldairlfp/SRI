from collections import Counter

import numpy as np
from documents import Document
from parse import Expression



from data_collections import default_processor

class Query(object):
    
    
    def precedence(token):
        """ Precedence of supported operators """
        __precedence = {"&": 2, "|": 1}
        try:
            return __precedence[token]
        except:
            return -1
    
    
    def is_left_bracket(token):
        """ Returns true if left bracket """
        return token == "("
    
    
    def is_right_bracket(token):
        """ Returns true if right bracket """
        return token == ")"
    
    
    def is_operator(token):
        """ Returns true if operator """
        return token == "&" or token == "|"
    
    
    def parse(self, query, processor: default_processor, lang: str = 'english'):
        """ Parse query into a list of tokens 
        :param self: self
        :type self: Query
        
        :param query: query to be parsed
        :type query: str
        
        :param processor: processor to be used
        :type processor: function
        
        :param lang: language of the query
        :type lang: str
        
        :return: list of tokens and boolean indicating if query is conjuntive
        :rtype : list, bool
        :return: processor, is_conjuntive
        """
        
        if '&' not in query:
            return processor(query, lang), False
        if '|' not in query:
            return processor(query, lang), True
            
        disjunctions = []
        current = 0
        index = 0
        rest = query
        is_conjuntive = False
        
        while(len(rest) > 0):
            if rest[index] == '(': 
                disjunctions.append(processor(query[current:index], lang))
                final = rest.findex(')')
                index = final - 1
                current = index + 2
                disjunctions.append(self.parse(query[current:index], processor, lang))
            elif rest[index] == ')':
                raise ValueError('Unbalanced brackets')
            elif rest[index] == '|':
                disjunctions.append(processor(query[current:index], lang))
                current = index + 1
                is_conjuntive = True
            index += 1
            rest = rest[current:]
            
        if len(rest) > 0:
            disjunctions.append(processor(query[current:], lang))
            
        return disjunctions, is_conjuntive        


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
            query_tfidf[word] = (
                self.a + (1 - self.a) * query_tfidf[word] / max_freq) * self._idf_dict[word]

        scores = [0 for doc in self.docs]
        for i, doc in enumerate(self.docs):
            for word in query_tfidf:
                if word in self._tf_idf[i]:
                    scores[i] += self._tf_idf[i][word] * query_tfidf[word] / np.sqrt(
                        sum([self._tf_idf[i][word] ** 2 for word in self._tf_idf[i]]))
        scores = np.array(scores)
        scores = scores.argsort()[-top:][::-1]
        return [self.docs[i] for i in scores]


class Extended(object):
    def __init__(self, docs):
        self.docs = docs
        self._tf_dict = self.term_frequency()
        self._idf_dict = self.inverse_doc_frequence()
        self._tf_idf = self.tf_idf(docs)
        self.vocabulary = self._idf_dict.keys()
        self._weights = self.weight()

        """
        Initialize an instace of the model
        :param docs: list of documents
        
        :type docs: Document
        :return: None
        """

    def term_frequency(self):
        tf_docs = [{} for doc in self.docs]

        """
        Calculate the term frequency for each word in each document
        
        :param self: instance of the model
        :type self: Extended
        
        :rtype: list
        :return: tf_docs
        """

        for doc in self.docs:
            for i, word in enumerate(doc.norm_corpus):
                if word in tf_docs[i]:
                    tf_docs[i][word] += 1
                else:
                    tf_docs[i][word] = 1

        return tf_docs

    def inverse_doc_frequence(self):
        df = {}

        """
        Calculate the inverse document frequency for each word in each document
        
        :param self: instance of the model
        :type self: Extended
        
        :rtype: dict
        :return: idf
        """

        for doc in self.docs:
            for word in doc.norm_corpus:
                if word in df:
                    df[word] += 1
                else:
                    df[word] = 1
        idf = {}
        for word in df:
            idf[word] = np.log10(len(self.docs) / df[word])
        return idf

    def tf_idf(self, doc: Document):
        tf_idf_scr = [{} for doc in self.docs]

        """
        Calculate the tf-idf score for each word in each document
        
        :param self: instance of the model
        :type self: Extended
        
        :param doc: list of documents
        :type doc: Document
        
        :rtype: list
        :return: tf_idf_scr
        """

        for i, doc in enumerate(self.docs):
            for word in doc.norm_corpus:
                try:
                    tf = self._tf_dict[i][word]
                    idf = self._idf_dict[word]
                    tf_idf_scr[i][word] = tf * idf
                except KeyError:
                    pass
        return tf_idf_scr

    def weight(self):
        """
        Calculate the weight of each word in each document

        :param self: instance of the model
        :type self: Extended

        :rtype: list
        :return: weight
        """

        weight = []
        for i, doc in enumerate(self.docs):
            weight.append([])
            for word in doc.norm_corpus:
                try:
                    weight[i].append({word : self.normalize_frequence(
                        self) * self._idf_dict[word] / max(self._idf_dict[word])})
                except KeyError:
                    weight[i].append({word : 0})
        return weight

    def normalize_frequence(self):
        """
        Normalize the frequency of each word in each document

        :param self: instance of the model
        :type self: Extended

        :rtype: list
        :return: normalize_frequence
        """

        normalize_frequence = []
        for i, doc in enumerate(self.docs):
            normalize_frequence.append([])
            for word in doc.norm_corpus:
                try:
                    normalize_frequence[i].append(
                        self._tf_dict[i][word] / len(doc.norm_corpus))
                except KeyError:
                    normalize_frequence[i].append(0)
        return normalize_frequence
        
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
        expr = Expression('or')
        expr.parse(query)
        
        # evaluate the query
        rank = expr.evaluate(self)
        
        # sort the documents based on the score
        rank = sorted(rank, key=lambda x: x[1], reverse=True)
        
        return rank[:10]
        
    

