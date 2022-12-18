from models import *
from data_collections import *

if __name__ == "__main__":
    # Create a welcome message
    print("Welcome to the SRI")
    print("=================================")
    path = 'D:/SRI/Dataset/cran.all.1400'
    print('Loading data...')
    with open(path, 'r', errors='ignore') as f:
        docs = CranCollection().parse(f)
    # vec = VectorSpace(docs)
    prob = Probabilistic(docs)

    while True:
        query = input("Enter your query: ")
        print()
        ranking = prob.get_ranking(query, 5)
        for i, doc in enumerate(ranking):
            print("Rank: ", i + 1)
            print("Doc ID: ", doc.id)
            print("Title: ", doc.title)
            print("Corpus: ", doc.corpus)
            print()
        
        print("Did you finish? (y/n)\n")
        ini = input()
        if ini == 'y':
            break
