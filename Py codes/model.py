#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import gensim
import numpy as np
from keras.models import Sequential
from keras.layers.recurrent import LSTM, SimpleRNN
from sklearn.model_selection import train_test_split
from keras import losses


with open('conversation.pickle', 'rb') as f:
    vec_x, vec_y = pickle.load(f)

vec_x = np.array(vec_x, dtype=np.float64)
vec_y = np.array(vec_y, dtype=np.float64)

x_train, x_test, y_train, y_test = train_test_split(vec_x, vec_y, test_size=0.2, random_state=1)


print("X_train_shape:", x_train.shape)
print("X_test_shape:", x_test.shape)


model = Sequential()
model.add(LSTM(output_dim=300,input_shape=x_train.shape[1:],return_sequences=True, init='glorot_normal', inner_init='glorot_normal', activation='tanh'))
model.add(LSTM(output_dim=300,input_shape=x_train.shape[1:],return_sequences=True, init='glorot_normal', inner_init='glorot_normal', activation='tanh'))
model.add(LSTM(output_dim=300,input_shape=x_train.shape[1:],return_sequences=True, init='glorot_normal', inner_init='glorot_normal', activation='tanh'))
model.compile(loss='cosine_proximity', optimizer='adam', metrics=['accuracy'])


print(model.summary())

model.fit(x_train, y_train, epochs=500,validation_data=(x_test, y_test))
model.save('LSTM500.h5');

model.fit(x_train, y_train, epochs=500,validation_data=(x_test, y_test))
model.save('LSTM1000.h5');

model.fit(x_train, y_train, epochs=500,validation_data=(x_test, y_test))
model.save('LSTM1500.h5');

model.fit(x_train, y_train, epochs=500,validation_data=(x_test, y_test))
model.save('LSTM2000.h5');

model.fit(x_train, y_train, epochs=500,validation_data=(x_test, y_test))
model.save('LSTM2500.h5');



