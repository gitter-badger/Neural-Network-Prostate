import Input
import Process

import tensorflow as tf


def train():
    with tf.Session() as sess:
        images, labels = Process.inputs()

        forward_propgation_results = Process.forward_propagation(images)

        train_loss = Process.error(forward_propgation_results, labels)

        init = tf.initialize_all_variables()

        sess.run(init)

def main(argv = None):
    train()

if __name__ == '__main__':
  tf.app.run()
