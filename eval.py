from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import main
import Process
import Input

FLAGS = tf.app.flags.FLAGS

checkpoint_dir = "/Users/Zanhuang/Desktop/NNP/Nerual-Network-Prostate.ckpt-0"
saver = tf.train.Saver()


def eval_once(top_k_op):
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
      print('No checkpoint file found')
      return

    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      true_count = 0
      total_sample_count = num_iter * FLAGS.batch_size
      step = 0
      while step < num_iter and not coord.should_stop():
        predictions = sess.run([top_k_op])
        true_count += np.sum(predictions)
        step += 1

      precision = true_count / total_sample_count
      print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate():
  with tf.Graph().as_default() as g:
    images, labels = Process.inputs()

    cost = Process.forward_propagation(images)

    top_k_op = tf.nn.in_top_k(cost, labels, 1)

    while True:
      eval_once(top_k_op)



def main(argv=None):

  evaluate()

if __name__ == '__main__':
  tf.app.run()
