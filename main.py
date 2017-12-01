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
  objects = tree.getroot().findall('object')
  annos = np.array([[float(x.text) for x in obj.find('bndbox').getchildren()] for obj in objects],np.float32)
  return [np.array([width,height]), annos]#,xmax,xmin,ymax,ymin])

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
  Images_train_filename = tf.string_join(
    [FLAGS.data_path,'Data','VID','train',ImageSets_train_prefix,'000000.JPEG'], separator='/')
  
  num_threads = 10
  batch_size = 100
  Annotations_train_filename_queue = tf.FIFOQueue(10000, dtypes=tf.string)
  Annotations_enqueue = Annotations_train_filename_queue.enqueue(Annotations_train_filename)
  tf.train.add_queue_runner(
          tf.train.QueueRunner(Annotations_train_filename_queue, [Annotations_enqueue] * num_threads))
  Images_train_filename_queue = tf.FIFOQueue(10000, dtypes=tf.string)
  Images_enqueue = Images_train_filename_queue.enqueue(Images_train_filename)
  tf.train.add_queue_runner(
          tf.train.QueueRunner(Images_train_filename_queue, [Images_enqueue] * num_threads))

  [Info_train, Bbox_train] = tf.py_func(read_anno_from_file, [Annotations_train_filename_queue.dequeue()], [tf.float32, tf.float32])
  
  image_reader = tf.WholeFileReader()
  _, image_file = image_reader.read(Images_train_filename_queue)
  image = tf.image.decode_jpeg(image_file)
  image = tf.image.convert_image_dtype(image, tf.float32)

  image = tf.Print(image, [ImageSets_train_filename_queue.size(), Annotations_train_filename_queue.size(), Images_train_filename_queue.size()])

  frame_queue = tf.FIFOQueue(1000, dtypes=[Bbox_train.dtype, image.dtype])

  frame_enqueue = frame_queue.enqueue([Bbox_train, image])
  tf.train.add_queue_runner(
          tf.train.QueueRunner(frame_queue, [frame_enqueue] * num_threads))

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(200):
      # Retrieve a single instance:
      [frame_anno, frame_image] = sess.run(frame_queue.dequeue())
      print i, frame_anno, frame_image.shape

    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
  tf.app.run()
