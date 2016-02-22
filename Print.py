import numpy as np
from numpy import genfromtxt

import tensorflow.python.platform
import tensorflow as tf

import Convolution

#EVALUATING DATA#
#STILL UNFINISHED

for x in range(100):
	y = tf.nn.softmax(tf.matmul(Convolution.x, Convolution.W) + Convolution.b)
	tf.sigmoid(y, name = None)
