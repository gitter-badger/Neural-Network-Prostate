import tensorflow as tf
import glob, os

IMAGE_SIZE = 1750
NUM_CLASSES = 4
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 100
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 100

def read_data(filename_queue):
  class CIFAR10Record(object):
    pass
  result = CIFAR10Record()

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

  print record_bytes

  return result

def generate_batch(image, label, min_queue_examples, batch_size, shuffle):
  num_preprocess_threads = 4
  images, label_batch = tf.train.batch([image, label], batch_size = batch_size, num_threads = num_preprocess_threads, capacity = min_queue_examples + 3 * batch_size)
  return images, tf.reshape(label_batch, [batch_size])

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  num_preprocess_threads = 16
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size = batch_size,
        num_threads = num_preprocess_threads,
        capacity = min_queue_examples + 3 * batch_size,
        min_after_dequeue = min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size = batch_size,
        num_threads = num_preprocess_threads,
        capacity = min_queue_examples + 3 * batch_size)

  return images, tf.reshape(label_batch, [batch_size])

def inputs(data_dir, batch_size):

  filenames = [os.path.join(data_dir, 'Prostate_Cancer_Data%d.binary' % i) for i in xrange(1, 4)]
  num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN

  filename_queue = tf.train.string_input_producer(filenames)

  read_input = read_data(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  print reshaped_image

  return _generate_image_and_label_batch(reshaped_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle = False)
