import Process

import time
import numpy as np
import os

import tensorflow as tf
from datetime import datetime

FLAGS = tf.app.flags.FLAGS


def train():
    with tf.Graph().as_default():

        images, labels = Process.inputs()

        forward_propgation_results = Process.forward_propagation(images)

        cost, train_loss = Process.error(forward_propgation_results, labels)             

        image_summary_t = tf.image_summary(images.name, images, max_images = 2)

        summary_op = tf.merge_all_summaries()

        init = tf.initialize_all_variables()

        saver = tf.train.Saver()
        
        sess = tf.InteractiveSession()        

        sess.run(init)

        saver = tf.train.Saver(tf.all_variables())
    
        tf.train.start_queue_runners(sess = sess)

        train_dir = "/home/zan/Desktop/Neural-Network-Prostate/model.ckpt"

        summary_writer = tf.train.SummaryWriter(train_dir, sess.graph)

        for step in xrange(650):
            start_time = time.time()
            print(sess.run([train_loss, cost]))
            duration = time.time() - start_time

            if step % 1 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, (%.1f examples/sec; %.3f ''sec/batch)')
                print (format_str % (datetime.now(), step, examples_per_sec, sec_per_batch))

                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)


            if step % 650 == 650:
                checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path)
                print "hello"

                sess.close()

                with tf.Session() as sess:                

                    eval_images, eval_labels = Process.eval_inputs()
        
                    evaluation_forward_propagation_results = Process.forward_propagation(eval_images)   
                    top_k_op = Process.evaluate(evaluation_forward_propagation_results, eval_labels)

                    for i in range(16):
                        print(sess.run([top_k_op]))


def main(argv = None):
    train()

if __name__ == '__main__':
  tf.app.run()
