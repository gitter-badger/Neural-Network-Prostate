import Process

import time
import numpy as np
import os

import tensorflow as tf
from six.moves import xrange
from datetime import datetime

FLAGS = tf.app.flags.FLAGS


def train():
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        images, labels = Process.inputs()
        forward_propgation_results = Process.forward_propagation(images)

        cost = Process.error(forward_propgation_results, labels)

        train_op = Process.train(cost, global_step)

        saver = tf.train.Saver()

        summary_op = tf.merge_all_summaries()

        init = tf.initialize_all_variables()

        sess = tf.InteractiveSession()

        sess.run(init)

        saver = tf.train.Saver(tf.all_variables())

        tf.train.start_queue_runners(sess = sess)

        train_dir = "/home/zan/nn-data"

        summary_writer = tf.train.SummaryWriter(train_dir, sess.graph)

        for step in xrange(650):
            start_time = time.time()
            _, loss_value = sess.run([train_op, cost])
            duration = time.time() - start_time

            assert not np.isnan(loss_value)

            if step % 1 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, (%.1f examples/sec; %.3f ''sec/batch)')
                print (format_str % (datetime.now(), step, examples_per_sec, sec_per_batch))

                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)


            if step % 20 or (step + 1) == 20:
                checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

def main(argv = None):
    train()

if __name__ == '__main__':
  tf.app.run()
