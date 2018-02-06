import tensorflow as tf
import argparse
import os
import sys
import csv
import glob
import logging


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='/Users/zhouyz/Development/ILSVRC2015')
parser.add_argument('--record_file', type=str, default='/Users/zhouyz/Development/ILSVRC2015/train.tfrecords')
parser.add_argument('--gpus', type=str, default='-1')
FLAGS, unparsed = parser.parse_known_args()
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpus

TASK = 'VID'
PHASES = ['train', 'test', 'val']

def main(_):
  record_reader = tf.TFRecordReader()
  record_queue = tf.train.string_input_producer([FLAGS.record_file])
  key, value = record_reader.read(record_queue)
  context, sequence_example = tf.parse_single_sequence_example(
    value,
    context_features={
      'snippet_id': tf.FixedLenFeature([], tf.string),
      'snippet_length': tf.FixedLenFeature([], tf.int64)
    },
    sequence_features={
      'frames': tf.FixedLenSequenceFeature([1], tf.int64),
      'bndboxes': tf.FixedLenSequenceFeature([4], tf.int64)
      # 'bndbox': tf.FixedLenSequenceFeature([4], tf.int64),
      # 'occluded': tf.FixedLenSequenceFeature([], tf.int64),
      # 'generated': tf.FixedLenSequenceFeature([], tf.int64)
    })
  # print(sequence_example['frameid'])
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in xrange(10):
      print sess.run([context['snippet_length'], sequence_example])
    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
  tf.app.run()