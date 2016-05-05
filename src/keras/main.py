# -*- coding: utf-8 -*-
# @Author: Chandan Yeshwanth
# @Date:   2016-05-01 16:33:49
# @Last Modified by:   Karthik
# @Last Modified time: 2016-05-06 00:47:56

import audio as aud

import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1234)

import time
import sys

from scipy.signal import resample, decimate

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM

from keras.models import model_from_json

DEFAULT_RATE = 4000

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

def get_data_labels(wav_data, seq_len=1000, shift=100):
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
	rate, data = aud.get_wav("violin_4k.wav")
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
	aud.save_wav(generated, "violin_gen.wav")

def generate_audio(X, Y, seed_X):
	"""
	X: array of input sequences 
	Y: next value for each input sequence
	seed_X: a single sequence to use as seed for generation
	"""
	# reshape to input format needed for the NN
	X = np.reshape(X, (X.shape[0], X.shape[1], 1))
	seed_X = np.reshape(seed_X, (seed_X.shape[0], seed_X.shape[1], 1))

	# train new model or use pre trained model?
	USE_SAVED_MODEL = False
	model_arch_file = 'model_architecture.json'
	model_weight_file = 'model_weights.h5'

	print "Architecture file:", model_arch_file
	print "Weight file:", model_weight_file

	model = None

	if USE_SAVED_MODEL:
		print "Loading model ..."
		model = model_from_json(open(model_arch_file).read())
		model.load_weights(model_weight_file)
	else:
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


		# save model
		print "Saving model ..."
		json_string = model.to_json()
		open(model_arch_file, 'w').write(json_string)
		model.save_weights(model_weight_file, overwrite=True)

	# compile model in both cases
	start = time.time()
	print "Started compilation: ", start
	model.compile(loss="mse", optimizer="rmsprop")
	print "Compilation Time: ", time.time() - start

	# train if using new model
	if not USE_SAVED_MODEL:
		# train
		model.fit(X, Y, 
			batch_size=32, 
			nb_epoch=5,
			validation_split=0.05
			)

	# generate new sequence
	model.reset_states()
	
	gen_seconds = 3
	generated = [None for i in range(DEFAULT_RATE) * gen_seconds]

	# generate 5 seconds of new music
	print seed_X.shape
	for i in xrange(DEFAULT_RATE * gen_seconds):
		sys.stdout.write("\r" + str(float(i)/(DEFAULT_RATE * gen_seconds)))
		predicted = model.predict(seed_X)[0]
		generated[i] = predicted
		seed_X[0,:,0] = np.append(seed_X[0,1:,0], predicted)
	
	return np.array(generated)

if __name__ == '__main__':
	main()