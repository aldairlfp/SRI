import numpy as np
import pickle
import os


def normalize_frequency(docs, tf_dict):
    """
    Normalize the frequency of each word in each document
    :param docs:
    :param tf_dict:
    :return:
    """

    normalize_frequency = []
    for i, doc in enumerate(docs):
        normalize_frequency.append([])
        for word in doc.norm_corpus:
            try:
                normalize_frequency[i].append(
                    tf_dict[i][word] / len(doc.norm_corpus))
            except KeyError:
                normalize_frequency[i].append(0)
    return normalize_frequency


def termFrequencyInDoc(docs):
    tf_docs = [{} for doc in docs]

    for i, doc in enumerate(docs):
        for word in doc.norm_corpus:
            if word in tf_docs[i]:
                tf_docs[i][word] += 1
            else:
                tf_docs[i][word] = 1

    return tf_docs


def wordDocFre(docs):
    df = {}
    for doc in docs:
        for word in doc.norm_corpus:
            if word in df:
                df[word] += 1
            else:
                df[word] = 1
    return df


def inverseDocFre(df_dict, length):
    idf = {}
    for word in df_dict:
        idf[word] = np.log10(length / df_dict[word])
    return idf


def tfidf(docs, tf_dict, idf_dict):
    tf_idf_scr = [{} for doc in docs]
    for i, doc in enumerate(docs):
        for word in doc.norm_corpus:
            try:
                tf = tf_dict[i][word]
                idf = idf_dict[word]
                tf_idf_scr[i][word] = tf * idf
            except KeyError:
                pass
    return tf_idf_scr


def serialize(data, path):
    with open(path, "wb") as outfile:
        pickle.dump(data, outfile)
    return data


def deserialize(path):
    with open(path, "rb") as infile:
        data = pickle.load(infile)
    return data


def explore_dir(corpus, r, p, docs):
    os.chdir(r)
    elements = os.listdir()
    for path in elements:
        n_p = p + '/' + path
        if os.path.isdir(path):
            explore_dir(corpus, path, n_p, docs)
            os.chdir('..')
        else:
            if not path.endswith('.txt'):
                docs.append('corpus/' + corpus + n_p)
    return docs
