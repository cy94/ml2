# -*- coding: utf-8 -*-
# @Author: Chandan Yeshwanth
# @Date:   2016-04-06 19:50:07
# @Last Modified by:   Chandan Yeshwanth
# @Last Modified time: 2016-04-06 22:27:16

import pandas as pd  
import numpy as np

from random import random

from scipy.io import wavfile

from keras.models import Sequential  
from keras.layers.core import Dense, Activation  
from keras.layers.recurrent import LSTM

SAMPLING_RATE = None

def get_data_from_wav(filename):
    r, d = wavfile.read(filename)
    right_stream = d[:,1]
    return r, right_stream

rate, data = get_data_from_wav("../../data/natabhairavi_violin.wav")
SAMPLING_RATE = rate

print data, len(data)
seconds = 5
# pick first few seconds
data = data[:SAMPLING_RATE*seconds]

pdata = pd.DataFrame({"a":data})  

def _load_data(data, n_prev = 20):  
    """
    data should be pd.DataFrame()
    """

    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data.iloc[i:i+n_prev].as_matrix())
        docY.append(data.iloc[i+n_prev].as_matrix())
    alsX = np.array(docX)
    alsY = np.array(docY)

    return alsX, alsY

def train_test_split(df, test_size=0.1):  
    """
    This just splits data to training and testing parts
    """
    ntrn = round(len(df) * (1 - test_size))
    print "splitting data"
    X_train, y_train = _load_data(df.iloc[0:ntrn])
    X_test, y_test = _load_data(df.iloc[ntrn:])
    print "split done"
    return (X_train, y_train), (X_test, y_test)

(X_train, y_train), (X_test, y_test) = train_test_split(pdata)  # retrieve data

in_out_neurons = 1
hidden_neurons = 20

model = Sequential()  

model.add(LSTM(hidden_neurons, input_dim=in_out_neurons, return_sequences=False))
model.add(Dense(in_out_neurons, input_dim=hidden_neurons))

model.add(Activation("linear"))  
model.compile(loss="mean_squared_error", optimizer="rmsprop")  


print "Fitting -"
model.fit(X_train, y_train, batch_size=700, nb_epoch=10, validation_split=0.05)  
print " fitted"

# generate new samples
old_sample = 100

for i in range(SAMPLING_RATE * 10):
	new_sample = model.predict(old_sample)  
	old_sample = new_sample

	print new_sample
