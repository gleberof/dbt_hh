import re
from multiprocessing import  Pool
from functools import partial
import numpy as np
import pandas as pd
import nltk

from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation
from gensim.models.doc2vec import Doc2Vec

#Create lemmatizer and stopwords list
mystem = Mystem() 
russian_stopwords = stopwords.words("russian")

regex = r"<((?=!\-\-)!\-\-[\s\S]*\-\-|((?=\?)\?[\s\S]*\?|((?=\/)\/[^.\-\d][^\/\]'\"[!#$%&()*+,;<=>?@^`{|}~ ]*|[^.\-\d][^\/\]'\"[!#$%&()*+,;<=>?@^`{|}~ ]*(?:\s[^.\-\d][^\/\]'\"[!#$%&()*+,;<=>?@^`{|}~ ]*(?:=(?:\"[^\"]*\"|'[^']*'|[^'\"<\s]*))?)*)\s?\/?))>"

reg_remove = r"<[^>]*>"

model = Doc2Vec.load('models/vector_size:400_min_count:2_epochs:20_window:5_seed:42_workers:4_negative:20.bin')

def extract_html(text):
    return [e[0] for e in re.findall(regex, text, re.MULTILINE)]

def remove_html(text):
    return re.sub(regex, '', text)

def get_similar(vect):
    return model.docvecs.most_similar([vect], topn=3)

def parallelize(data, func, num_of_processes=12, result_format='df'):
    data_split = np.array_split(data, num_of_processes)
    pool = Pool(num_of_processes)
    if result_format == 'df':
        data = pd.concat(pool.map(func, data_split))
    else:
        data = pool.map(func, data_split)
    pool.close()
    pool.join()
    return data

def run_on_subset(func, data_subset):
    return data_subset.apply(func)

def parallelize_on_rows(data, func, num_of_processes=12, result_format = 'df'):
    return parallelize(data, partial(run_on_subset, func), num_of_processes, result_format)


def preprocess_text(text):
    tokens = mystem.lemmatize(text.lower())
    tokens = [token.strip() for token in tokens if token not in russian_stopwords\
              and token != " " \
              and token.strip() not in punctuation
             and len(token.strip()) > 2
             and token.strip().isalnum()
             and not token.strip().isnumeric()]
    
    text = " ".join(tokens)
    
    return text
