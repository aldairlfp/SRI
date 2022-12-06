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


class Binary(object):

    def __init__(self):
        self.terms = []  # list to store the terms present in the documents
        self.keys = []  # list to store the names of the documents
        self.vec_Dic = {}  # dictionary to store the name of the document and the boolean vector as list
        self.dicti = {}  # dictionary to store the name of the document and the terms present in it as a vector
        self.dummy_List = []  # list for performing some operations and clearing them

    def filter(self, documents, rows, cols):
        """function to read and separate the name of the documents and the terms
        present in it to a separate list  from the data frame and also create a
        dictionary which has the name of the document as key and the terms present in
        it as the list of strings  which is the value of the key"""

        for i in range(rows):
            for j in range(cols):
                # traversal through the data frame

                if j == 0:
                    # first column has the name of the document in the csv file
                    self.keys.append(documents.loc[i].iat[j])
                else:
                    self.dummy_List.append(documents.loc[i].iat[j])
                    # dummy list to update the terms in the dictionary

                    if documents.loc[i].iat[j] not in self.terms:
                        # add the terms to the list if it is not present else continue
                        self.terms.append(documents.loc[i].iat[j])

            copy = self.dummy_List.copy()
            # copying the dummy list to a different list

            self.dicti.update({documents.loc[i].iat[0]: copy})
            # adding the key value pair to a dictionary

            self.dummy_List.clear()
            # clearing the dummy list

    def bool_Representation(self, rows, cols):
        """In this function we get a boolean representation of the terms present in the
        documents in the form of lists, later we create a dictionary which contains
        the name of the documents as key and value as the list of boolean values
        representing the terms present in the document"""

        self.terms.sort()
        # we sort the elements in the alphabetical order for the convenience, the order
        # of the term does not make any difference

        for i in self.dicti:
            # for every document in the dictionary we check for each string present in
            # the list

            for j in self.terms:
                # if the string is present in the list we append 1 else we append 0

                if j in self.dicti[i]:
                    self.dummy_List.append(1)
                else:
                    self.dummy_List.append(0)
                # appending 1 or 0 for obtaining the boolean representation

            copy = self.dummy_List.copy()
            # copying the dummy list to a different list

            self.vec_Dic.update({i: copy})
            # adding the key value pair to a dictionary

            self.dummy_List.clear()
            # clearing the dummy list

    def query_Vector(self, query):
        """In this function we represent the query in the form of boolean vector"""

        qvect = []
        # query vector which is returned at the end of the function

        for i in self.terms:
            # if the word present in the list of terms is also present in the query
            # then append 1 else append 0

            if i in query:
                qvect.append(1)
            else:
                qvect.append(0)

        return qvect
        # return the query vector which is obtained in the boolean form

    def prediction(self, q_Vect):
        """In this function we make the prediction regarding which document is related
        to the given query by performing the boolean operations"""

        dictionary = {}
        listi = []
        count = 0
        # initialisation of the dictionary , list and a variable which is further
        # required for performing the computation

        term_Len = len(self.terms)
        # number of terms present in the term list

        for i in self.vec_Dic:
            # for every document in the dictionary containing the terms present in it
            # the form of boolean vector

            for t in range(term_Len):
                if q_Vect[t] == self.vec_Dic[i][t]:
                    # if the words present in the query is also present in the
                    # document or if the words present in the query is also absent in
                    # the document

                    count += 1
                    # increase the value of count variable by one
                    # the condition in which words present in document and absent in
                    # query , present in query and absent in document is not considered

            dictionary.update({i: count})
            # dictionary update here the name of the document is the key and the
            # count variable computed earlier is the value

            count = 0
            # reinitialization of count variable to 0

        for i in dictionary:
            listi.append(dictionary[i])
            # here we append the count value to list

        listi = sorted(listi, reverse=True)
        # we sort the list in the descending order which is needed to rank the
        # documents according to the relevance

        ans = ' '
        # variable to store the name of the document which is most relevant

        print("ranking of the documents")

        for count, i in enumerate(listi):

            key = Binary.get_key(i, dictionary)
            # Function call to get the key when the value is known
            if count == 0:
                ans = key
                # to store the name of the document which is most relevant

            print(key, "rank is", count + 1)
            # print the name of the document along with its rank

            dictionary.pop(key)
            # remove the key from the dictionary after printing

            print(ans, "is the most relevant document for the given query")
            # to print the name of the document which is most relevant

    @staticmethod
    def get_key(val, dictionary):
        """
        function to get the key from the value
        :param dictionary: dictionary
        :param val: value
        :return: key
        """
        for key, value in dictionary.items():
            if val == value:
                return key

        return "key doesn't exist"
