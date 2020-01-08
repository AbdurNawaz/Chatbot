#!/usr/bin/env python
# coding: utf-8

import os
import json
import nltk
from gensim import models
import numpy as np
from gensim import corpora, models, similarities
import pickle
import numpy as np

model = models.Word2Vec.load('word2vec.bin')

with open('conversation.json', 'r') as f:
    data = json.load(f)
    cor = data["conversations"]

#Seperating Questions and answers into x and y respectively
x = []
y = []

for i in range(len(cor)):
    for j in range(len(cor[i])-1):
        x.append(cor[i][j])
        y.append(cor[i][j+1])


print("X:\n", x[:10])

print("Y:\n", y[:10])


#Tokenizing all the sentences in the dataset 
tok_x = []
tok_y = []

for i in range(len(x)):
    tok_x.append(nltk.word_tokenize(x[i].lower()))
    tok_y.append(nltk.word_tokenize(y[i].lower()))

print("Tokenized X:\n"tok_x[:10])


#Removing punctuation characters and converting each word into vector using word2vec model

vec_x = []
vec_y = []
from string import punctuation

for sent in tok_x:
    sentvec = [w for w in sent if w not in punctuation]
    sentvec1 = [model[w] for w in sentvec if w in model.wv.vocab]    
    vec_x.append(sentvec1)

for sent in tok_y:
    sentvec = [w for w in sent if w not in punctuation]
    sentvec1 = [model[w] for w in sentvec if w in model.wv.vocab]    
    vec_y.append(sentvec1)

#Clipping each sentence to length 15 and padding with a vector of
#ones to th sentences with length<15

sentend = np.ones((300,), dtype=np.float32)

for tok_sent in vec_x:
    tok_sent[14: ] =[]
    tok_sent.append(sentend)
    
for tok_sent in vec_x:
    if(len(tok_sent)<15):
        for i in range(15-len(tok_sent)):
            tok_sent.append(sentend)
            
for tok_sent in vec_y:
    tok_sent[14: ] =[]
    tok_sent.append(sentend)
    
for tok_sent in vec_y:
    if(len(tok_sent)<15):
        for i in range(15-len(tok_sent)):
            tok_sent.append(sentend)

#Saving the preprocessed data into a pickle file
with open('conversation.pickle','wb') as f:
    pickle.dump([vec_x,vec_y],f)
