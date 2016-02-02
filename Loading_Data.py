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
	
	data = tf.image.decode_jpeg(filename, channels=1, ratio=None, fancy_upscaling=None, 	try_recover_truncated=None, acceptable_fraction=None, name= 'Decoded Jpeg')
	
	tf.train.match_filenames_once('Gleason 3', name = "GLEASON 3")
	
	print data
	
os.chdir("Gleason 4 Cribiform")


for filename in glob.glob("*.jpg"):
	
	
	img = Image.open(filename)
	
	data2 = tf.image.decode_jpeg(filename, channels=1, ratio=None, fancy_upscaling=None, 	try_recover_truncated=None, acceptable_fraction=None, name= 'Decoded Jpeg')
	
	tf.train.match_filenames_once('Gleason 4 Cribiform', name = "GLEASON 4 CRIBIFORM")
	
	print data2
	
	
os.chdir("Gleason 4 Non Cribiform")

for filename in glob.glob("*.jpg"):
	
	img = Image.open(filename)
	
	data3 = tf.image.decode_jpeg(filename, channels=1, ratio=None, fancy_upscaling=None, 	try_recover_truncated=None, acceptable_fraction=None, name= 'Decoded Jpeg')
	
	tf.train.match_filenames_once('Gleason 4 Cribiform', name = "GLEASON 4 CRIBIFORM")
	
	print data3

os.chdir("Normal Prostate")

for filename in glob.glob("*.jpg"):
	
	
	img = Image.open(filename)
	
	data4 = tf.image.decode_jpeg(filename, channels=1, ratio=None, fancy_upscaling=None, 	try_recover_truncated=None, acceptable_fraction=None, name= 'Decoded Jpeg')
	
	tf.train.match_filenames_once('Gleason 4 Cribiform', name = "GLEASON 4 CRIBIFORM")
	
	print data4

