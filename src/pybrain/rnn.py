# -*- coding: utf-8 -*-
# @Author: Chandan Yeshwanth
# @Date:   2016-04-06 10:08:49
# @Last Modified by:   Chandan Yeshwanth
# @Last Modified time: 2016-04-06 11:21:45

from __future__ import print_function

from pybrain.datasets import SequentialDataSet

from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import LSTMLayer

from pybrain.supervised import RPropMinusTrainer

from sys import stdout
from itertools import cycle

def get_data_from_wav(filename):
	return []

def main():
	data = get_data_from_wav()

	ds = SequentialDataSet(1, 1)
	for sample, next_sample in zip(data, cycle(data[1:])):
	    ds.addSample(sample, next_sample)

	net = buildNetwork(1, 5, 1, 
                   hiddenclass=LSTMLayer, outputbias=False, recurrent=True)

	trainer = RPropMinusTrainer(net, dataset=ds)
	train_errors = [] # save errors for plotting later
	EPOCHS_PER_CYCLE = 5
	CYCLES = 100
	EPOCHS = EPOCHS_PER_CYCLE * CYCLES
	for i in xrange(CYCLES):
	    trainer.trainEpochs(EPOCHS_PER_CYCLE)
	    train_errors.append(trainer.testOnData())
	    epoch = (i+1) * EPOCHS_PER_CYCLE
	    print("\r epoch {}/{}".format(epoch, EPOCHS), end="")
	    stdout.flush()

	# predict new values
	old_sample = 100

	for i in xrange(10000):
		new_sample = net.activate(old_sample)
		old_sample = new_sample

		print new_sample
		
if __name__ == '__main__':
	main()











