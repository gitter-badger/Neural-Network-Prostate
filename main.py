import Input
import Process

import tensorflow as tf

def train():
    with tf.Session() as sess:
        images, labels = Process.inputs()

        forward_propgation_results = Process.forward_propagation(images)

        train_loss, cost = Process.error(forward_propgation_results, labels)

        init = tf.initialize_all_variables()

        tf.train.start_queue_runners(sess=sess)

        sess.run(init)

        for i in range(1000):
            print(sess.run([train_loss, cost]))

def main(argv = None):
    train()

if __name__ == '__main__':
  tf.app.run()
