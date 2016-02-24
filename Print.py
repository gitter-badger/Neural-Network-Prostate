import numpy as np
from numpy import genfromtxt

import tensorflow.python.platform
import tensorflow as tf

import Convolution
import Loading_Data

#EVALUATING DATA#
#STILL UNFINISHED

for x in range(100):
	y = tf.nn.softmax(tf.matmul(Convolution.x, Convolution.W) + Convolution.b)
	with tf.Session():
  		input = tf.placeholder(tf.float32)
  		classifier = Loading_Data.Gleason_3
  		print classifier.eval(feed_dict={input: my_python_preprocessing_fn()})
