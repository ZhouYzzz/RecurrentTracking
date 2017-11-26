#!/usr/bin/python

import numpy as np
import tensorflow as tf
import os, sys

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("data_path", '/home/spark/data/ILSVRC2015/ILSVRC2015',
                    "Where the training/test/val data is stored.")
FLAGS = flags.FLAGS

def main(_):
  print tf.gfile.Exists(FLAGS.data_path)
  ImageSets_train_filename = tf.constant(
    [os.path.join(FLAGS.data_path,'ImageSets','VID','train%d.txt'%(i)) for i in xrange(1,31)])
  ImageSets_val_filename = tf.constant(
    os.path.join(FLAGS.data_path,'ImageSets','VID','val.txt'))
  ImageSets_test_filename = tf.constant(
    os.path.join(FLAGS.data_path,'ImageSets','VID','test.txt'))
  ImageSets_train_filename_queue = tf.train.string_input_producer(ImageSets_train_filename)
  reader = tf.TextLineReader()
  key, value = reader.read(ImageSets_train_filename_queue)
  record_defaults = [['/path/prefix/to/record'],[1]]
  ImageSets_train_prefix, _ = tf.decode_csv(
    value, record_defaults=record_defaults)

  with tf.Session() as sess:
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  for i in range(1200):
    # Retrieve a single instance:
    prefix = sess.run(ImageSets_train_prefix)
    print i, prefix

  coord.request_stop()
  coord.join(threads)

if __name__ == '__main__':
  tf.app.run()
