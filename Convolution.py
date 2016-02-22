import numpy as np
from numpy import genfromtxt

import tensorflow.python.platform
import tensorflow as tf

import Loading_Data

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape = [None, 4900])
y_ = tf.placeholder(tf.float32, shape = [None, 4])
W = tf.Variable(tf.zeros([4900, 4]))
b = tf.Variable(tf.zeros([4]))

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)

def conv2d(x, w):
	return tf.nn.conv2d(x, w, strides = [5, 5, 1, 1], padding = 'SAME')
	#After conv2d there is 122500 matrix.

def max_pool_5x5(x):
	return tf.nn.max_pool(x, ksize = [1, 5, 5, 1], strides = [1, 1, 1, 1], padding = 'SAME')
	#After the max_pool there is 4900 matrix.
