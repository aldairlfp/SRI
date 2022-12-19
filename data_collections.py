from nltk.corpus import stopwords
from nltk import word_tokenize
import string
# stemming
from nltk.stem import PorterStemmer
import os

from documents import Document


def default_processor(raw_text, language):
    """This method has the proposal of processing the text
    :param raw_text: The text to be processed
    :param language: The language of the text
    :return: A List of stemmed and normalized tokens
    """
    result = word_tokenize(raw_text, language)
    stop_list = stopwords.words(language) + [*string.punctuation]
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in result if (token not in stop_list) and len(token) >= 3]


class Collection:

    def __init__(self, corpus: str, processor=default_processor, lang: str = 'english'):
        """
        Collection initializer
        :param processor: Text processor to use in the collection
        :param lang: Language of the collection
        """
        self._corpus = corpus
        self._processor = processor
        self._lang = lang
        docs = []
        self.explore_dir('corpus/' + corpus, '', docs)
        self._pre_docs = docs
        os.chdir('../..')

    def explore_dir(self, r, p, docs):
        os.chdir(r)
        elements = os.listdir()
        for path in elements:
            n_p = p + '/' + path
            if os.path.isdir(path):
                self.explore_dir(path, n_p, docs)
                os.chdir('..')
            else:
                if not path.endswith('.txt'):
                    docs.append('corpus/' + self._corpus + n_p)
        return docs

    def parse(self):
        raise NotImplemented()


class CranCollection(Collection):

    def __init__(self, processor=default_processor, lang='english'):
        super().__init__('cran')

    def parse(self):
        file = open(self._pre_docs[0], 'r')
        docs = []
        doc_id = 0
        text = ''
        subject = ''
        in_subject = 0
        in_text = 0
        while True:
            line = file.readline()
            if len(line.split()) > 0:
                if line.split()[0] == '.A':
                    in_subject = 0
                    line = file.readline()
                    text += line
                elif line.split()[0] == '.B':
                    line = file.readline()
                    text += line
                elif line.split()[0] == '.I':
                    if in_text:
                        doc = Document(doc_id, subject, text, self._processor, self._lang)
                        docs.append(doc)
                        doc_id += 1
                        in_text = 0
                        subject = ''
                        text = ''
                    # doc_id = int(line.split()[1])
                elif line.split()[0] == '.T':
                    in_subject = 1
                elif line.split()[0] == '.W':
                    in_text = 1
                elif in_text:
                    text += line
                elif in_subject:
                    subject += line
            elif not line:
                doc = Document(doc_id, subject, text, self._processor, self._lang)
                docs.append(doc)
                doc_id += 1
                break
        file.close()
        return docs


class NewsGroupCollection(Collection):
    def __init__(self, processor=default_processor, lang='english'):
        super().__init__('newsgroup')

    def _get_document(self, file):
        text = ''
        subject = ''
        s = 1
        while True:
            line = file.readline()
            if not line:
                break
            text += line
            if s and line.split()[0] == 'Subject:':
                s = 0
                subject = ' '.join(line.split()[1:])
        return [subject, text]

    def parse(self):
        docs = []
        for path in self._pre_docs:
            file = open(path, 'r')
            subject, text = self._get_document(file)
            doc = Document(path, subject, text, self._processor, self._lang)
            docs.append(doc)
            file.close()
        return docs
