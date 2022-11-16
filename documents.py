class Document:
    """This is a Document object containing the original Document and the normalized version of it.
    """

    def __init__(self, id, title, corpus, processor, lang):
        self.id = id
        self.title = title
        self.corpus = corpus
        self.lang = lang
        self.norm_title = processor(title, lang)
        self.norm_corpus = processor(corpus, lang)
