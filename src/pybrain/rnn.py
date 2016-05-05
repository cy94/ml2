# -*- coding: utf-8 -*-
# @Author: Chandan Yeshwanth
# @Date:   2016-04-06 10:08:49
# @Last Modified by:   Karthik
# @Last Modified time: 2016-04-06 12:07:50

from __future__ import print_function

from pybrain.datasets import SequentialDataSet

from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import LSTMLayer

from pybrain.supervised import RPropMinusTrainer

from sys import stdout
from itertools import cycle


from scipy.io import wavfile
import numpy as np
# numpy.set_printoptions(threshold=numpy.nan)



def get_data_from_wav(filename):
	r, d = wavfile.read(filename)
	right_stream = d[:,1]
	return r, right_stream


def main():
	generated_data = [0 for i in range(10000)]
	rate, data = get_data_from_wav("../../data/natabhairavi_violin.wav")
	data = data[1000:190000]
	print("Got wav")
	ds = SequentialDataSet(1, 1)
	for sample, next_sample in zip(data, cycle(data[1:])):
	    ds.addSample(sample, next_sample)

	net = buildNetwork(1, 5, 1, 
                   hiddenclass=LSTMLayer, outputbias=False, recurrent=True)

	trainer = RPropMinusTrainer(net, dataset=ds)
	train_errors = [] # save errors for plotting later
	EPOCHS_PER_CYCLE = 5
	CYCLES = 50
	EPOCHS = EPOCHS_PER_CYCLE * CYCLES
	for i in xrange(CYCLES):
	    trainer.trainEpochs(EPOCHS_PER_CYCLE)
	    train_errors.append(trainer.testOnData())
	    epoch = (i+1) * EPOCHS_PER_CYCLE
	    print("\r epoch {}/{}".format(epoch, EPOCHS), end="")
	    stdout.flush()

	# predict new values
	old_sample = 100

	for i in xrange(500000):
		new_sample = net.activate(old_sample)
		old_sample = new_sample
		generated_data[i] = new_sample[0]
		print(new_sample)
	
	wavfile.write("../../output/test.wav",  rate, np.array(generated_data))

if __name__ == '__main__':
	main()












