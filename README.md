# Information Retrieval System

## Objective

The objective of this work is to carry out an information retrieval system
that allows users to perform searches in a set of documents,
and obtain as a result a list of documents ordered by relevance

## Description

The information retrieval system that will be developed in this work
it would be an information retrieval system based on three methods of
recover of information:

- Information recovery method based on the Vectorial Space Model
- Information recovery method based on the probabilistic model
- Information retrieval method based on the extended Boolean model

The user is allowed to decide if he wants to use any of the three methods
to perform information recovery. The user must provide a collection of documents, and a search query.

## Use

To use the application you just have to run the file `main.py` with python. You should also create two folders in the root directory: *data* and *models*, this is used to save the collections and models respectively in serialized form to optimize runtime

## Requirements

You must have `python`>=3.7 installed and install the modules in *requirements.txt* by running the command `pip install -r requirements.txt`. Finally, the necessary words to execute the **NLTK** module must be downloaded, for this we execute the following python code:

```python
import nltk

nltk.download('punkt')
nltk.download('stopwords')
```

## Collections

The collections must be inside a folder called corpus, inside the root directory and inside there must be a folder for each type of collection, for now only the three:

- Cran
- Reuters
- Newsgroup
