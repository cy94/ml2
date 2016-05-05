# -*- coding: utf-8 -*-
# @Author: Chandan Yeshwanth
# @Date:   2016-05-01 16:33:49
# @Last Modified by:   Chandan Yeshwanth
# @Last Modified time: 2016-05-05 17:56:14

import audio as aud

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import resample, decimate

from keras.models import Sequential

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
	new_data = aud.cut_wav(new_data, 10, 15)
	print "Cut:", new_data.size
	aud.save_wav(new_data, "violin_cut.wav")

	# plt.hist(new_data)
	# plt.show()

	X, Y = get_data_labels(new_data)
	generated = generate_audio(X, Y)

	aud.save_wav(new_data, "violin_cut.wav")

def generate_audio(X, Y):
	model = Sequential()

	# add layers
	# model.add(LSTM(32, stateful=True))
	# model.add(Dropout(0.2))
	# model.add(Activation('sigmoid'))

	# compile model

	# train
	# model.fit(data, labels)

	# generate
	
	return np.array(0)

if __name__ == '__main__':
	main()