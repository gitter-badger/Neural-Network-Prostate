from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import Input
import main
import Process

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/home/zan/nn-data',
                           """Directory where to read model checkpoints.""")

def eval_once(saver, top_k_op):
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(16 / 1))
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter
      step = 0
      while step < num_iter and not coord.should_stop():
        predictions = sess.run([top_k_op])
        true_count += np.sum(predictions)
        step += 1

      # Compute precision @ 1.
      precision = true_count / total_sample_count
      print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate():
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    eval_data = FLAGS.eval_data == 'test'
    images, labels = Process.eval_inputs()

    prediction = Process.forward_propagation(images)

    top_k_op = tf.nn.in_top_k(prediction, labels, 1)

    variable_averages = tf.trainable_variables()
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)


    while True:
      eval_once(saver, top_k_op)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):
  evaluate()


if __name__ == '__main__':
  tf.app.run()









"""
import tensorflow as tf

import main
import Process
import Input

eval_dir = "/tmp/nn-data"
checkpoint_dir = "/tmp/nn-data"


def evaluate():
  with tf.Graph().as_default() as g:
    images, labels = Process.eval_inputs()
    forward_propgation_results = Process.forward_propagation(images)
    init_op = tf.initialize_all_variables()
    saver = tf.train.Saver()
    top_k_op = tf.nn.in_top_k(forward_propgation_results, labels, 1)

  with tf.Session(graph = g) as sess:
    tf.train.start_queue_runners(sess = sess)
    sess.run(init_op)
    #saver.restore(sess, eval_dir)
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    	if ckpt and ckpt.model_checkpoint_path:
     	 # Restores from checkpoint
     	 saver.restore(sess, ckpt.model_checkpoint_path)
      	# Assuming model_checkpoint_path looks something like:
      	#   /my-favorite-path/cifar10_train/model.ckpt-0,
      	# extract global_step from it.
      	global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    	else:
     	 print('No checkpoint file found')
         return

    for i in range(100):
        print(sess.run(top_k_op))

def main(argv = None):
    evaluate()

if __name__ == '__main__':
    tf.app.run()
"""
