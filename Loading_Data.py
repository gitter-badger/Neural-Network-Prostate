from PIL import Image

import numpy as np
from numpy import genfromtxt
import os, glob

import tensorflow.python.platform
import tensorflow as tf

print "ENCODING IMAGES IN AN UINT8 MATRIX FILE"
print "WARNING: FILE FORMAT MUST BE .jpg or .jpeg."

os.chdir('DATA_PREPARE/Gleason 3')

for filename in glob.glob("*.jpg"):

	img = Image.open(filename)

	data = tf.image.decode_jpeg(filename, channels = 1, ratio = None, fancy_upscaling = None, 	try_recover_truncated = None, acceptable_fraction = None, name = 'Decoded Jpeg')

	tf.train.match_filenames_once('Gleason 3', name = "GLEASON_3")

	print data

	training_data = filename
	training_labels = "Gleason_3"

	with tf.Session():
	  input_data = tf.constant(training_data)
	  input_labels = tf.constant(training_labels)


os.chdir("Gleason 4 Cribiform")


for filename in glob.glob("*.jpg"):


	img = Image.open(filename)

	data2 = tf.image.decode_jpeg(filename, channels = 1, ratio = None, fancy_upscaling = None, try_recover_truncated = None, acceptable_fraction = None, name = 'Decoded Jpeg')

	tf.train.match_filenames_once('Gleason 4 Cribiform', name = "GLEASON_4_CRIBIFORM")

	print data2

	training_data = filename
	training_labels = "Gleason_4_Cribiform"

	with tf.Session():
	  input_data = tf.constant(training_data)
	  input_labels = tf.constant(training_labels)


os.chdir("Gleason 4 Non Cribiform")

for filename in glob.glob("*.jpg"):

	img = Image.open(filename)

	data3 = tf.image.decode_jpeg(filename, channels = 1, ratio = None, fancy_upscaling = None, 	try_recover_truncated = None, acceptable_fraction = None, name = 'Decoded Jpeg')

	tf.train.match_filenames_once('Gleason 4 Non Cribiform', name = "GLEASON_4_NON_CRIBIFORM")

	print data3

	training_data = filename
	training_labels = "Gleason_4_Non_Cribiform"

	with tf.Session():
		input_data = tf.constant(training_data)
		input_labels = tf.constant(training_labels)

os.chdir("Normal Prostate")

for filename in glob.glob("*.jpg"):


	img = Image.open(filename)

	data4 = tf.image.decode_jpeg(filename, channels = 1, ratio = None, fancy_upscaling = None, 	try_recover_truncated = None, acceptable_fraction = None, name = 'Decoded Jpeg')

	tf.train.match_filenames_once('Normal Prostate', name = "NORMAL_PROSTATE")

	training_data = filename
	training_labels = "Gleason_4_Non_Cribiform"

	with tf.Session():

		input_data = tf.constant(training_data)

		input_labels = tf.constant(training_labels)

	print data4
