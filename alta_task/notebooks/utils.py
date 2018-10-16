import io
import pandas as pd
import spacy

from sklearn.model_selection import cross_val_score


nlp = spacy.load('en', disable=['ner', 'parser'])


def get_train_scores(clf, train_text, train_labels):
    scores = cross_val_score(clf, train_text, train_labels, cv=5, scoring='f1_micro', n_jobs=4)
    return scores


def normalize_text(path):
    with open(path, encoding='iso8859-1') as file:
        text = file.read()
    text = text[:999999]
    return '\n'.join(tok.lemma_ for tok in nlp(text) if not (tok.is_space or tok.is_punct))