# -*- coding: utf-8 -*-
# @Author: Chandan Yeshwanth
# @Date:   2016-05-01 16:33:49
# @Last Modified by:   Chandan Yeshwanth
# @Last Modified time: 2016-05-05 21:02:12

import audio as aud

import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1234)

import time

from scipy.signal import resample, decimate

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM

DEFAULT_RATE = 44100

def roundup(x, up=100):
	return x if x % up == 0 else x + up - x % up

def quantize(signal, partitions, codebook):
	""" http://stackoverflow.com/questions/15983986/convert-quantiz-function-to-python """
	indices = []
	quanta = []
	for datum in signal:
		index = 0
		while index < len(partitions) and datum > partitions[index]:
		    index += 1
		indices.append(index)
		quanta.append(codebook[index])
	return np.array(quanta)

def get_data_labels(wav_data, seq_len=50, shift=5):
	""" create sequences of seq_len with a shift of shift values"""

	new_data = np.copy(wav_data)
	new_data -= int(new_data.mean())

	result = []

	for index in range(0, len(new_data) - seq_len, shift):
		result.append(new_data[index: index + seq_len])


	data = np.array(result)
	# input - all but the last column 
	X = data[:, :-1]
	# output - last column 
	Y = data[:,  -1]

	return X, Y

def main():
	rate, data = aud.get_wav("violin.wav")
	print "Original:", data.size

	new_data = np.copy(data)

	# select 5 seconds of audio
	train_data = aud.cut_wav(new_data, 10, 15)
	print "Train:", train_data.size
	aud.save_wav(train_data, "violin_train.wav")

	seed_data = aud.cut_wav(new_data, 16, 17)

	X, Y = get_data_labels(train_data)
	seed_X, seed_Y = get_data_labels(seed_data)

	generated = generate_audio(X, Y, np.array([seed_X[0]]))
	# aud.save_wav(generated, "violin_gen.wav")

def generate_audio(X, Y, seed_X):
	"""
	X: array of input sequences 
	Y: next value for each input sequence
	seed_X: a single sequence to use as seed for generation
	"""
	# reshape to input format needed for the NN
	X = np.reshape(X, (X.shape[0], X.shape[1], 1))
	seed_X = np.reshape(seed_X, (seed_X.shape[0], seed_X.shape[1], 1))

	model = Sequential()
	layers = [1, 10, 20, 1]

	# add layers
	model.add(LSTM(
            input_dim=layers[0],
            output_dim=layers[1],
            return_sequences=True,
            # stateful=True,
            # batch_input_shape=(32, 49, 1)
            ))
	model.add(Dropout(0.2))

	model.add(LSTM(
            layers[2],
            return_sequences=False,
            # stateful=True,
            # batch_input_shape=(32, 49, 1)
            ))
	model.add(Dropout(0.2))

	model.add(Dense(
            output_dim=layers[3]))
	model.add(Activation("linear"))

	# compile model
	start = time.time()
	print "Started compilation: ", start
	model.compile(loss="mse", optimizer="rmsprop")
	print "Compilation Time: ", time.time() - start

	# train
	model.fit(X, Y, 
		batch_size=32, 
		nb_epoch=1,
		validation_split=0.05
		)

	# generate
	generated = []
	model.reset_states()

	# generate 5 seconds of new music
	for i in xrange(DEFAULT_RATE * 5):
		predicted = model.predict(seed_X)[0]
		print predicted
		generated.append(predicted)
		seed_X = np.append(seed_X[1:], predicted)
	
	return generated

if __name__ == '__main__':
	main()