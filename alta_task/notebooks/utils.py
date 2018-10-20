import io
import multiprocessing as mp
from functools import partial
from pathlib import Path
from itertools import zip_longest

import pandas as pd
import spacy

from sklearn.model_selection import cross_val_score


nlp = spacy.load('en', disable=['ner', 'parser'])


def get_train_scores(clf, train_texts, train_labels):
    scores = cross_val_score(clf, train_texts, train_labels, cv=5, scoring='f1_micro', n_jobs=4)
    return scores


def normalize_text(path):
    with open(path, encoding='iso8859-1') as file:
        text = file.read()
    text = text[:999999]
    return ' '.join(tok.lemma_ for tok in nlp(text) if not tok.is_punct)


def token_cleaner(tokens):
    for token in tokens:
        if token.like_num:
            yield f'_NUMBER_'
        elif token.is_currency:
            yield f'_CURRENCY_'
        else:
            yield token.lower_


def clean_text(input_path, output_dir=''):
    with open(input_path, encoding='iso8859-1') as file:
        text = file.read()

    text = text[:999999]
    preprocessed = (tok for tok in nlp(text) if not (tok.is_punct or tok.is_space or tok.is_stop))
    normalised = token_cleaner(preprocessed)
    output = ' '.join(tok for tok in normalised)
    output_path = Path(output_dir) / Path(input_path).name 

    print(output_path)
    with open(str(output_path), 'w') as file:
        file.write(output)

        
def clean_files(input_dir, output_dir, workers=1, limit=None):
    input_paths = sorted(Path(input_dir).glob('*'))

    if limit is not None:
        input_paths = input_paths[:limit]
        
    if workers == 1:
        for path in input_paths:
            clean_text(path, output_dir=output_dir)
    else:
        func = partial(clean_text, output_dir=output_dir)
        with mp.Pool(processes=workers) as pool:
            list(pool.imap_unordered(func, input_paths))
            
            
class MeanEmbeddingVectorizer:
    """Taken from: http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/"""
    def __init__(self, embeddings, dim=300):
        self.embeddings = embeddings
        self.dim = dim

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.embeddings[w] for w in words if w in self.embeddings]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

    
def load_fasttext_vectors(fname):
    with open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
        n, d = map(int, fin.readline().split())    
        data = {}
        for line in fin:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = map(float, tokens[1:])
    return data