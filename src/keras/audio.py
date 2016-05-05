# -*- coding: utf-8 -*-
# @Author: Chandan Yeshwanth
# @Date:   2016-05-01 16:37:07
# @Last Modified by:   Karthik
# @Last Modified time: 2016-05-06 00:34:05

from scipy.io import wavfile
import os
import numpy as np

DATA_DIR = "../../data/"
RATE = 4000

def get_wav(filename):
	""" return wav data """
	rate, data = wavfile.read(os.path.join(DATA_DIR, filename))
	right_stream = data[:,1]
	return rate, np.array(right_stream)

def save_wav(data, filename, rate=RATE):
	""" write wav data to file """
	path = os.path.join(DATA_DIR, filename)
	wavfile.write(path, rate, data)
	print "Wrote:", path

def divide_wav(wav, num_parts):
	pass

def average_wav(data, interval, rate=RATE):
	""" find the average of every 'interval' seconds
	from: http://stackoverflow.com/questions/15956309/averaging-over-every-n-elements-of-a-numpy-array"""

	window_size = int(rate*interval)
	new_data = np.copy(data)

	if new_data.size % window_size != 0:
		# remove elements to make size of array divisible by window_size
		new_data = new_data[:-(new_data.size % window_size)]

	return np.mean(new_data.reshape(-1, window_size), axis=1)

def cut_wav(data, start_time, end_time, rate=RATE):
	""" select part of wav from start to end time"""

	return data[rate*start_time: rate*end_time]

def expand_wav(data, n):
	""" repeat every element n times """

	return np.repeat(data, n)
