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
        self._pre_docs = self.explore_dir('corpus/' + corpus, '')

    def explore_dir(self, r, p):
        docs = []
        os.chdir(r)
        elements = os.listdir()
        for path in elements:
            if os.path.isdir(path):
                n_p = p + '/' + path
                self.explore_dir(path, n_p)
                os.chdir('..')
            else:
                file = open(path, 'r', errors='ignore')
                docs.append(file)
                file.close()
        return docs

    def parse(self):
        raise NotImplemented()


class CranCollection(Collection):

    def __init__(self, processor=default_processor, lang='english'):
        super().__init__(processor, lang)

    def parse(self, file):
        docs = []
        id = 0
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
                        doc = Document(id, subject, text, self.processor, self.lang)
                        docs.append(doc)
                        in_text = 0
                        subject = ''
                        text = ''
                    doc_id = int(line.split()[1])
                elif line.split()[0] == '.T':
                    in_subject = 1
                elif line.split()[0] == '.W':
                    in_text = 1
                elif in_text:
                    text += line
                elif in_subject:
                    subject += line
            elif not line:
                doc = Document(id, subject, text, self.processor, self.lang)
                docs.append(doc)
                break

        return docs


class NewsGroupCollection(Collection):
    def __init__(self, processor=default_processor, lang='english'):
        super().__init__(processor, lang)

    def get_document(self, file):
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
