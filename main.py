#!/usr/bin/python

import numpy as np
import tensorflow as tf

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("data_path", '/home/spark/data/ILSVRC2015/ILSVRC2015',
                    "Where the training/test/val data is stored.")
FLAGS = flags.FLAGS

def main(_):
  print tf.gfile.Exists(FLAGS.data_path)

if __name__ == '__main__':
  tf.app.run()
