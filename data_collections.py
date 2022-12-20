from nltk.corpus import stopwords
from nltk import word_tokenize
import string
# stemming
from nltk.stem import PorterStemmer
import os

from documents import Document
import utils


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
        self.docs = []
        docs = []
        utils.explore_dir(self._corpus, 'corpus/' + corpus, '', docs)
        self._pre_docs = docs
        os.chdir('../..')

    def save(self):
        utils.serialize(self.docs, 'data/' + self._corpus + '_docs.pkl')

    def load(self):
        self.docs = utils.deserialize('data/' + self._corpus + '_docs.pkl')


class CranCollection(Collection):

    def __init__(self, processor=default_processor, lang='english'):
        super().__init__('cran')
        try:
            self.load()
        except FileNotFoundError:
            file = open(self._pre_docs[0], 'r')
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
                            self.docs.append(doc)
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
                    self.docs.append(doc)
                    doc_id += 1
                    break
            file.close()
            self.save()


class NewsGroupCollection(Collection):
    def __init__(self):
        super().__init__('newsgroup')
        try:
            self.load()
        except FileNotFoundError:
            for i, path in enumerate(self._pre_docs):
                file = open(path, 'r')
                subject, text = self._get_document(file)
                doc = Document(i, subject, text, self._processor, self._lang)
                self.docs.append(doc)
                file.close()
            self.save()

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

    def save(self):
        utils.serialize(self.docs, 'data/newsgroup_docs.pkl')
        return self.docs

    def load(self):
        self.docs = utils.deserialize('data/newsgroup_docs.pkl')
        return self.docs


class ReutersParser(Collection):
    def init(self):
        super().__init__("reuters")
        try:
            self.load()
        except FileNotFoundError:
            for i, path in enumerate(self._pre_docs):
                file = open(path, 'r')
                docs = self._get_document(file)
                self.docs.extend(docs)
            self.save()

    def _get_document(self, file):
        docs = []
        doc_id = 0
        text = ""
        title = ""
        in_text = 0
        while True:
            line = file.readline()
            if len(line) > 0:
                if line.find("<REUTERS") != -1:
                    doc_id = int(line.split("NEWID=")[1].split('"')[1])
                elif line.find("<TITLE>") != -1:
                    title = line.split("<TITLE>")[1].split("</TITLE>")[0]
                elif line.find("<BODY>") != -1:
                    in_text = 1
                    text += line.split("<BODY>")[1]
                elif line.find("</BODY>") != -1:
                    in_text = 0
                    doc = Document(
                        doc_id, title, text, self._processor, self._lang
                    )
                    docs.append(doc)
                    text = ""
                    title = ""
                elif in_text:
                    text += line
            elif not line:
                break
        file.close()
        return docs
