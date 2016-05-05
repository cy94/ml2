# -*- coding: utf-8 -*-
# @Author: Chandan Yeshwanth
# @Date:   2016-05-01 16:33:49
# @Last Modified by:   Chandan Yeshwanth
# @Last Modified time: 2016-05-03 22:01:30

import audio as aud

import numpy as np
import matplotlib.pyplot as plt


DEFAULT_RATE = 44100

def roundup(x, up=100):
	return x if x % up == 0 else x + up - x % up

def main():
	rate, data = aud.get_wav("violin.wav")
	print "Original:", data.size

	new_data = np.copy(data)

	# select 5 seconds of audio
	new_data = aud.cut_wav(new_data, 10, 15)
	print "Cut:", new_data.size
	aud.save_wav(new_data, "violin_cut.wav")

	THRESHOLD = 8000

	new_data = np.vectorize(roundup)(new_data)

	plt.hist(new_data)
	plt.show()
	# interval = 0.0001

	# new_data = aud.average_wav(new_data, interval)
	# print "Averaged:", new_data.size

	# new_data = aud.expand_wav(new_data, DEFAULT_RATE * interval)
	# print "Expanded:", new_data.size

	aud.save_wav(new_data, "violin_new.wav")

if __name__ == '__main__':
	main()