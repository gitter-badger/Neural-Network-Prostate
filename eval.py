import tensorflow as tf

import main
import Process
import Input

eval_dir = "/Users/Zanhuang/Desktop/NNP/model.ckpt-250"
checkpoint_dir = "/Users/Zanhuang/Desktop/NNP/checkpoint"


def evaluate():
  with tf.Graph().as_default() as g:
    images, labels = Process.eval_inputs()
    forward_propgation_results = Process.forward_propagation(images)
    init_op = tf.initialize_all_variables()
    saver = tf.train.Saver()
    top_k_op = tf.nn.in_top_k(forward_propgation_results, labels, 1)

  with tf.Session(graph = g) as sess:
    sess.run(init_op)
    tf.train.start_queue_runners(sess=sess)
    saver.restore(sess, eval_dir)
    for i in range(100):
        print(sess.run(top_k_op))

def main(argv = None):
    evaluate()

if __name__ == '__main__':
  tf.app.run()
