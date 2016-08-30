from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import glob, os

from six.moves import xrange

IMAGE_SIZE = 1750
NUM_CLASSES = 4
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 4
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1000


def read_data(filename_queue):
  class Record(object):
    pass
  result = Record()

  label_bytes = 1
  result.height = 1750
  result.width = 1750
  result.depth = 3

  image_bytes = result.height * result.width * result.depth

  record_bytes = label_bytes + image_bytes

  reader = tf.FixedLengthRecordReader(record_bytes = record_bytes)

  result.key, value = reader.read(filename_queue)

  record_bytes = tf.decode_raw(value, tf.uint8)

  result.label = tf.cast(tf.slice(record_bytes, [0], [label_bytes]), tf.int32)

  depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]), [result.depth, result.height, result.width])

  result.uint8image = tf.transpose(depth_major, [1, 2, 0])

  return result

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  num_preprocess_threads = 8
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  return images, tf.reshape(label_batch, [batch_size])


def inputs(data_dir, batch_size):
  filenames = [os.path.join(data_dir, 'Prostate_Cancer_Data%d' % i)
               for i in xrange(1, 4)]
  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  filename_queue = tf.train.string_input_producer(filenames)

  read_input = read_data(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  min_fraction_of_examples_in_queue = 1
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)

  print ('Filling queue with %d CIFAR images before starting to train. This will take a few minutes.' % min_queue_examples)

  return _generate_image_and_label_batch(reshaped_image, read_input.label, min_queue_examples, batch_size, shuffle=True)

def eval_inputs(data_dir, batch_size):
  filenames = [os.path.join(data_dir, 'Prostate_Cancer_Data1.bin')]
  num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  filename_queue = tf.train.string_input_producer(filenames)

  read_input = read_data(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  min_fraction_of_examples_in_queue = 1
  min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)

  return _generate_image_and_label_batch(reshaped_image, read_input.label, min_queue_examples, 1, shuffle=False)
