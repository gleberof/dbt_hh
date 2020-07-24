import telepot
bot = telepot.Bot('436447719:AAGq_iK2hE3cPFhmL9Jh53g9HyVkw3SXZbg')
def qq(txt):
    bot.sendMessage(1114926, txt)
    
    
import json
import gzip
from itertools import islice
from collections import Counter, defaultdict
from operator import itemgetter

import dill
import os
    
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.callbacks import CallbackAny2Vec

import random

import argparse

import time
import datetime

parser = argparse.ArgumentParser(description='Doc2Vec params')

parser.add_argument('--vector_size', type=int, help='vector_size')
parser.add_argument('--min_count', type=int, help='min_count')
parser.add_argument('--epochs', type=int, help='epochs')
parser.add_argument('--window', type=int, help='window')
parser.add_argument('--seed', type=int, help='seed')
parser.add_argument('--workers', type=int, help='seed')
parser.add_argument('--negative', type=int, help='negative')

class ProgressMeter(CallbackAny2Vec):
    def __init__(self, epochs):
        self.epoch = 0
        self.time = time.time()
        self.start = self.time
        self.epochs = epochs
    def on_epoch_begin(self, model):
        print(f'Start epoch {self.epoch}')
        self.time = time.time()
        if self.epoch == 0:
            self.start = self.time

    def on_epoch_end(self, model):
        self.epoch += 1
        new_time = time.time()
        
        est_time = datetime.timedelta(seconds = (new_time - self.start) * (self.epochs - self.epoch) / self.epoch)
        if self.epochs == self.epoch:
            print(f'Epoch  takes {datetime.timedelta(seconds = new_time - self.time)}. ' \
                  f'Total time: {datetime.timedelta(seconds = new_time - self.start)}')
        else:
            print(f'Epoch  takes {datetime.timedelta(seconds = new_time - self.time)}. Estimate to end {est_time}')
        self.time = new_time
        
def train_model(document_text, word_freq = None, corpus_count=None, **kwargs):
    prms = {}
    for key, value in kwargs.items():
        if value is not None:
            prms[key] = value
    
    print('Preparing DOC2VEC model:', ', '.join([f'{k}:{v}' for k,v in prms.items()]))
    model = Doc2Vec(**prms)
    if word_freq is None:
        model.build_vocab(document_text)
    else:
        model.build_vocab_from_freq(word_freq, corpus_count=corpus_count)
    print('Vocab ready')
    
    print(model.docvecs.vectors_docs.shape)
        
    model.train(document_text, total_examples=model.corpus_count, epochs=model.epochs, callbacks=[ProgressMeter(model.epochs)])
    
    model.save(os.path.join('models', '_'.join([f'{k}:{v}' for k,v in prms.items()]) + '.bin'))
    
    print('Model ready '+', '.join([f'{k}:{v}' for k,v in prms.items()]))
    

    num_samples = 1000
    TOPN = 10
    doc_samples = random.sample(range(len(document_text)),num_samples)
    ranks = []
    for doc_id in doc_samples:
        inferred_vector = model.infer_vector(document_text[doc_id].words)
        sims = model.docvecs.most_similar([inferred_vector], topn=TOPN)
        try:
            ranks.append([docid for docid, sim in sims].index(document_text[doc_id].tags[0]))
        except:
            ranks.append(TOPN+1)
    
    # qq(counter[0]/num_samples)
    print(Counter(ranks)[0]/num_samples, Counter(ranks)[TOPN+1]/num_samples)
    qq(f'{Counter(ranks)[0]/num_samples}, {Counter(ranks)[TOPN+1]/num_samples}')
         

if __name__ == '__main__':
    
    args = parser.parse_args()
    
    document_text = dill.load(open('tagged_docs.bin', 'rb'))
    
    train_model(vector_size=args.vector_size, min_count=args.min_count, 
                epochs=args.epochs, window=args.window, seed=args.seed, 
                workers=args.workers, negative=args.negative, document_text=document_text)

