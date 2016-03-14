import tensorflow as tf
import numpy as np

import imageflow
import os, glob

sess = tf.InteractiveSession()

os.chdir("/Users/Zanhuang/Desktop/NNP/DATA")


#def read_csv(filename_and_label_tensor):
	
	#filename, label = tf.decode_csv(filename_and_label_tensor, [[""], [""]], " ")
	#file_contents = tf.read_file(filename)
	#example = tf.image.decode_jpeg(file_contents)
	#return example, label
	
#read_csv("file_input.csv")
#########################################
#import tensorflow as tf

#filenames = ['/Users/Zanhuang/Desktop/NNP/DATA']
#filename_queue = tf.train.string_input_producer(filenames)

#reader = tf.WholeFileReader()
#key, value = reader.read(filename_queue)

#images = tf.image.decode_jpeg(value, channels=1)

#print images

#path = '/Users/Zanhuang/Desktop/NNP/DATA'

	
#filename_queue = tf.train.string_input_producer(path)

#reader = tf.WholeFileReader()
#key, value = reader.read(filename_queue)

#images = tf.image.decode_jpeg(value, channels = 1)


def read_jpeg(filename_queue):
 reader = tf.WholeFileReader()
 key, value = reader.read(filename_queue)

 my_img = tf.image.decode_jpeg(value) 
 my_img.set_shape([1750, 1750, 1])
 return my_img


def read_image_data():
 jpeg_files = []
 images_tensor = []

 i = 1
 WORKING_PATH = "/Users/Zanhuang/Desktop/NNP/DATA"
 jpeg_files_path = glob.glob(os.path.join(WORKING_PATH, '*.jpeg'))

 for filename in jpeg_files_path:
 	print(i)
 	i += 1
 	jpeg_files.append(filename)

 # Create a queue that produces the filenames to read
 filename_queue = tf.train.string_input_producer(jpeg_files)

 mlist = [read_jpeg(filename_queue) for _ in range(len(jpeg_files))]  # This list h as all the images decoded correctly

 init = tf.initialize_all_variables()

 sess = tf.Session()
 sess.run(init)

 coord = tf.train.Coordinator()  # I started the queues
 threads = tf.train.start_queue_runners(sess=sess, coord=coord)

 images_tensor = tf.convert_to_tensor(images_tensor)

 sess.close()

read_image_data()