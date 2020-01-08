#!/usr/bin/env python
# coding: utf-8

import numpy as np
import nltk
import gensim
from keras.models import load_model
import warnings
warnings.simplefilter("ignore")


#Loading our saved model
model=load_model("LSTM1000.h5")


mod = gensim.models.Word2Vec.load('word2vec.bin')


while True:
    x=input("Enter the message: ")
    sentend=np.ones((300,),dtype=np.float32) 

    sent=nltk.word_tokenize(x.lower())
    sentvec = [mod[w] for w in sent if w in mod.wv.vocab]

    sentvec[14:]=[]
    sentvec.append(sentend)
    if len(sentvec)<15:
        for i in range(15-len(sentvec)):
            sentvec.append(sentend) 
    sentvec=np.array([sentvec])
    
    predictions = model.predict(sentvec)
    outputlist=[mod.most_similar([predictions[0][i]])[0][0] for i in range(15)]
    r = ["kleiser", "karluah", "post-oscar", "ballets"]
    out = [outputlist[i] for i in range(15) if outputlist[i] not in r]
    output=' '.join(out)
    print("BOT:", output)


