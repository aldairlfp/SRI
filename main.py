from models import *
from data_collections import *

if __name__ == "__main__":
    # Create a welcome message
    print("Welcome to the SRI")
    print("=================================")
    path = 'corpus/cran/cran.all.1400'
    print('Loading data...')
    try:
        vec = utils.deserialize('models/boolean_extended.pkl')
    except FileNotFoundError:
        docs = CranCollection().parse()
        vec = BooleanExtended(docs)
        utils.serialize(vec, 'models/boolean_extended.pkl')

    while True:
        query = input("Enter your query: ")
        if query == 'q':
            break
        print()
        ranking = vec.ranking(query, 5)
        for i, doc in enumerate(ranking):
            print("Rank: ", i + 1)
            print("Doc ID: ", doc.id)
            print("Title: ", doc.title)
            print("Corpus: ", doc.corpus)
            print()