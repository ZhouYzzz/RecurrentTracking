#!/usr/bin/python

import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET
import os, sys

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("data_path", '/home/spark/data/ILSVRC2015/ILSVRC2015',
                    "Where the training/test/val data is stored.")
FLAGS = flags.FLAGS

def read_anno_from_file(filename):
  tree = ET.parse(filename)
  size = tree.getroot().find('size')
  width = np.float32(size.find('width').text)
  height = np.float32(size.find('height').text)
  bndbox = tree.getroot().find('object').find('bndbox')
  xmax = np.float32(bndbox.find('xmax').text)
  xmin = np.float32(bndbox.find('xmin').text)
  ymax = np.float32(bndbox.find('ymax').text)
  ymin = np.float32(bndbox.find('ymin').text)
  return np.array([width,height,xmax,xmin,ymax,ymin])

def main(_):
  print tf.gfile.Exists(FLAGS.data_path)
  ImageSets_train_filename = tf.constant(
    [os.path.join(FLAGS.data_path,'ImageSets','VID','train_%d.txt'%(i)) for i in xrange(1,31)])
  ImageSets_val_filename = tf.constant(
    os.path.join(FLAGS.data_path,'ImageSets','VID','val.txt'))
  ImageSets_test_filename = tf.constant(
    os.path.join(FLAGS.data_path,'ImageSets','VID','test.txt'))
  ImageSets_train_filename_queue = tf.train.string_input_producer(ImageSets_train_filename)
  reader = tf.TextLineReader()
  key, value = reader.read(ImageSets_train_filename_queue)
  record_defaults = [['/path/prefix/to/record'],[1]]
  ImageSets_train_prefix, _ = tf.decode_csv(
    value, record_defaults=record_defaults, field_delim=' ')
  Annotations_train_filename = tf.string_join(
    [FLAGS.data_path,'Annotations','VID','train',ImageSets_train_prefix,'000000.xml'], separator='/')
  Annotations_train = tf.py_func(read_anno_from_file, [Annotations_train_filename], tf.float32)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(120):
      # Retrieve a single instance:
      filename, anno = sess.run([Annotations_train_filename, Annotations_train])
      print i, filename, anno

    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
  tf.app.run()
