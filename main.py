from models import *
from data_collections import *
import gui


def main():
    # Create a welcome message
    print("Welcome to the SRI")
    print("=================================")
    path = 'D:/SRI/Dataset/cran.all.1400'
    print('Loading data...')
    collection = NewsGroupCollection()
    collection2 = CranCollection()
    collection3 = ReutersCollection()
    print('Data loaded')


if __name__ == "__main__":
    main()
