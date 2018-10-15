#!/usr/bin/env python
# coding: utf-8

# In[20]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import naive_bayes
from sklearn.svm import LinearSVC

plt.style.use('fivethirtyeight')
#from spacy.lang.en.stop_words import STOP_WORDS


# In[16]:


train = pd.read_csv('../input/alta2018traindata/train_data.csv')
train['filename'] = ['../input/alta2018patents/patents/patents/'] + train['id'].astype(str) + ['.txt']

test = pd.read_csv('../input/alta2018testdata/test_data.csv')


# In[17]:


train


# In[24]:


tfidf = TfidfVectorizer(
    input='filename', 
    encoding='iso8859-1',
    stop_words='english',
    strip_accents='unicode',
    ngram_range=(1,2)
)

svm_clf_ngram2= Pipeline(steps=[
    ('tfidf', tfidf), 
    ('svm', LinearSVC(class_weight='balanced'))
])


# Train and evaluate:

# In[21]:


scores = cross_val_score(
    svm_clf, train['filename'], train['first_ipc_mark_section'], cv=5, scoring='f1_micro', n_jobs=4
)
scores.mean()


# In[ ]:


scores = cross_val_score(
    svm_clf_ngram2, train['filename'], train['first_ipc_mark_section'], cv=5, scoring='f1_micro', n_jobs=1
)
scores.mean()

