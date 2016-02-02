from PIL import Image
from numpy import array
import glob
import os

import tensorflow.python.platform

import tensorflow as tf


os.chdir("DATA_PREPARE/Gleason 3")

for filename in glob.glob("*.jpg"):
	img = Image.open(filename)
	width, height = img.size
	
	data_size = width * height
	
	kernel = 81
	padding = 0
	
	
	num_neurons = data_size - kernel + 2 * padding
	
	sess = tf.InteractiveSession()
	
	print "Hidden Neurons = %d\n" % num_neurons
	print "Starting to Prepare Data for Usage\n"
	print "Convoluting .........."
		
	init_op = tf.initialize_all_variables()
	
	sess.run(init_op)
		
	
	
	