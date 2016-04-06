# -*- coding: utf-8 -*-
# @Author: Karthik
# @Date:   2016-04-05 19:04:03
# @Last Modified by:   Karthik
# @Last Modified time: 2016-04-06 11:24:46

from scipy.io import wavfile
# import numpy
# numpy.set_printoptions(threshold=numpy.nan)


r, d = wavfile.read("../data/natabhairavi_violin.wav")
right_stream = d[:,1]
print ' '.join(map(str, list(right_stream)[10000:20000]))
wavfile.write("../output/test.wav",  r, right_stream)