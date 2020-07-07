import re
from multiprocess import  Pool
from functools import partial
import numpy as np
import pandas as pd
import nltk

from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation

#Create lemmatizer and stopwords list
mystem = Mystem() 
russian_stopwords = stopwords.words("russian")

regex = r"<((?=!\-\-)!\-\-[\s\S]*\-\-|((?=\?)\?[\s\S]*\?|((?=\/)\/[^.\-\d][^\/\]'\"[!#$%&()*+,;<=>?@^`{|}~ ]*|[^.\-\d][^\/\]'\"[!#$%&()*+,;<=>?@^`{|}~ ]*(?:\s[^.\-\d][^\/\]'\"[!#$%&()*+,;<=>?@^`{|}~ ]*(?:=(?:\"[^\"]*\"|'[^']*'|[^'\"<\s]*))?)*)\s?\/?))>"

reg_remove = r"<[^>]*>"

def extract_html(text):
    return [e[0] for e in re.findall(regex, text, re.MULTILINE)]

def remove_html(text):
    return re.sub(regex, '', text)


def parallelize(data, func, num_of_processes=12):
    data_split = np.array_split(data, num_of_processes)
    pool = Pool(num_of_processes)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data

def run_on_subset(func, data_subset):
    return data_subset.apply(func)

def parallelize_on_rows(data, func, num_of_processes=12):
    return parallelize(data, partial(run_on_subset, func), num_of_processes)


def preprocess_text(text):
    tokens = mystem.lemmatize(text.lower())
    tokens = [token for token in tokens if token not in russian_stopwords\
              and token != " " \
              and token.strip() not in punctuation]
    
    text = " ".join(tokens)
    
    return text