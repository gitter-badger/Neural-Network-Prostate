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
	
	

	filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once("*.jpg"))
	image_reader = tf.WholeFileReader()
	_, image_file = image_reader.read(filename_queue)

	image = tf.image.decode_jpeg(image_file)
	
	print image
	


os.chdir("Gleason 4 Cribiform")


for filename in glob.glob("*.jpg"):


	filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once("*.jpg"))
	image_reader = tf.WholeFileReader()
	_, image_file = image_reader.read(filename_queue)

	image = tf.image.decode_jpeg(image_file)

	print image


os.chdir("Gleason 4 Non Cribiform")

for filename in glob.glob("*.jpg"):

	filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once("*.jpg"))
	image_reader = tf.WholeFileReader()
	_, image_file = image_reader.read(filename_queue)

	image = tf.image.decode_jpeg(image_file)

	print image

os.chdir("Normal Prostate")

for filename in glob.glob("*.jpg"):


	filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once("*.jpg"))
	image_reader = tf.WholeFileReader()
	_, image_file = image_reader.read(filename_queue)

	image = tf.image.decode_jpeg(image_file)

	print image