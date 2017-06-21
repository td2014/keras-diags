#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 12:12:49 2017

@author: anthonydaniell
"""
# Start: Set up environment for reproduction of results
import numpy as np
import tensorflow as tf
import random as rn
import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
#single thread
session_conf = tf.ConfigProto(
      intra_op_parallelism_threads=1,
      inter_op_parallelism_threads=1)

from keras import backend as K
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
# End:  Set up environment for reproduction of results

#
from keras.layers import LSTM, Dense, Embedding
from keras.models import Sequential
from keras.preprocessing import sequence

#
# Create input sequences
#
word_index=21
train_data_new = []
train_data_new.append([1, 1, 1, 1, 1, 1, 1])
train_data_new.append([2, 2, 2, 2, 2, 2, 2])
train_data_new.append([2, 2, 2, 2, 2, 2, 2])
train_data_new.append([1, 1, 1, 1, 1, 1, 1])
train_data_new.append([2, 2, 2, 2, 2, 2, 2])
train_data_new.append([2, 2, 2, 2, 2, 2, 2])
train_data_new.append([1, 1, 1, 1, 1, 1, 1])
train_data_new.append([2, 2, 2, 2, 2, 2, 2])
train_data_new.append([2, 2, 2, 2, 2, 2, 2])

#
# Preprocess
#

max_length = 7
X_train = sequence.pad_sequences(train_data_new, maxlen=max_length, padding='post')

# preparing y_train
y_train = []
y_train.append([1,0])
y_train.append([0,1])
y_train.append([0,1])
y_train.append([1,0])
y_train.append([0,1])
y_train.append([0,1])
y_train.append([1,0])
y_train.append([0,1])
y_train.append([0,1])

y_train = np.array(y_train)

#
# Create model
#

EMBEDDING_DIM=16

model = Sequential()
model.add(Embedding(word_index + 1, EMBEDDING_DIM, input_length=max_length))
model.add(LSTM(3))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print(model.summary())

#
# Train
# 
print('Training model...')
model.fit(X_train, y_train, epochs=30)

#
# output predictions
#
print(model.get_weights()[0][0])
predictions = model.predict(X_train)

#
# End of script
#
