import os
import sys

import tensorflow as tf
import Input

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 20, "hello")
tf.app.flags.DEFINE_string('data_dir', '/Volumes/Machine_Learning_Data/DATA', "hello")

def inputs():
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'Data')
  images, labels = Input.inputs(data_dir = data_dir, batch_size = FLAGS.batch_size)
  return images, labels

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def conv2d(images, W):
    return tf.nn.conv2d(images, W, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool_5x5(images):
    return tf.nn.max_pool(images, ksize = [1, 5, 5, 1], strides = [1, 5, 5, 1], padding = 'SAME')

def forward_propagation(images):
  with tf.variable_scope('conv1') as scope:
      W_conv1 = weight_variable([5, 5, 3, 32])
      b_conv1 = bias_variable([32])
      image_matrix = tf.reshape(images, [-1, 1750, 1750, 3])
      h_conv1 = tf.nn.sigmoid(conv2d(image_matrix, W_conv1) + b_conv1)
      h_pool1 = max_pool_5x5(h_conv1)

  with tf.variable_scope('conv2') as scope:
      W_conv2 = weight_variable([5, 5, 32, 64])
      b_conv2 = bias_variable([64])
      h_conv2 = tf.nn.sigmoid(conv2d(h_pool1, W_conv2) + b_conv2)
      h_pool2 = max_pool_5x5(h_conv2)

  with tf.variable_scope('conv3') as scope:
      W_conv3 = weight_variable([5, 5, 64, 128])
      b_conv3 = bias_variable([128])
      h_conv3 = tf.nn.sigmoid(conv2d(h_pool2, W_conv3) + b_conv3)
      h_pool3 = max_pool_5x5(h_conv3)

  with tf.variable_scope('local3') as scope:
      W_fc1 = weight_variable([14 * 14 * 128, 256])
      b_fc1 = bias_variable([256])
      h_pool3_flat = tf.reshape(h_pool3, [FLAGS.batch_size, 14 * 14 * 128])
      h_fc1 = tf.nn.sigmoid(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
      keep_prob = tf.placeholder(tf.float32)
      h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
      W_fc2 = weight_variable([256, 4])
      b_fc2 = bias_variable([4])

      y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
      return y_conv

def error(forward_propagation_results, labels):
    labels = tf.one_hot(labels, 4)
    tf.transpose(labels)
    labels = tf.cast(labels, tf.float32)
    mean_squared_error = tf.square(tf.sub(labels, forward_propagation_results))
    cost = tf.reduce_mean(mean_squared_error)
    train = tf.train.GradientDescentOptimizer(learning_rate = 0.3).minimize(cost)
    return train, cost
