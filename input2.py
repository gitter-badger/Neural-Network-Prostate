import tensorflow as tf
import numpy as np

import imageflow
import os, glob

sess = tf.InteractiveSession()

def read_jpeg(filename_queue):
 reader = tf.WholeFileReader()
 key, value = reader.read(filename_queue)

 my_img = tf.image.decode_jpeg(value)
 my_img.set_shape([1750, 1750, 1])
 print(value)
 return my_img

#####################################################
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


 filename_queue = tf.train.string_input_producer(jpeg_files)

 mlist = [read_jpeg(filename_queue) for _ in range(len(jpeg_files))]

 init = tf.initialize_all_variables()

 sess = tf.Session()
 sess.run(init)


 images_tensor = tf.convert_to_tensor(images_tensor)
 

 batch = tf.train.batch(mlist, 100, num_threads=4, capacity=100, enqueue_many=True)
 print batch

 print(sess.run(batch))

 coord.join(threads)

 sess.close()

