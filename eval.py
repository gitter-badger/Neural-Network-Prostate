
import tensorflow as tf

import main
import Process
import Input

eval_dir = "/home/zan/Desktop/Neural-Network-Prostate/model.ckpt"
checkpoint_dir = "/home/zan/Desktop/Neural-Network-Prostate/model.ckpt"
#ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

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
    saver.restore(sess, eval_dir)
    #if ckpt and ckpt.model_checkpoint_path:
      #  saver.restore(sess, ckpt.model_checkpoint_path)

    for i in range(100):
        print(sess.run(top_k_op))

def main(argv = None):
    evaluate()

if __name__ == '__main__':
    tf.app.run()
