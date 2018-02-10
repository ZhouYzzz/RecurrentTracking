import tensorflow as tf
import os
import argparse
import glob
import logging
import csv
import itertools
from typing import List, Dict
# helper functions
from snippet import Snippet, Shot, SnippetRegister
from features import * # feature_* and feature_list_*
from annotations import parse_annotation_file

parser = argparse.ArgumentParser()
parser.add_argument('--record_dir', type=str, default='/tmp')
parser.add_argument('--data_dir', type=str, default='/Users/zhouyz/Development/ILSVRC2015')
parser.add_argument('--phase', type=str, default='train')
FLAGS = parser.parse_args()

logging.basicConfig(level=logging.INFO)

def get_snippet_list_files(data_path: str, phase: str = 'train'):
  pattern = os.path.join(data_path, 'ImageSets', 'VID', '{}*.txt'.format(phase))
  return glob.glob(pattern)

def parse_snippet(datapath: str, snippet_id: str, phase: str = 'train'):
  annotation_path = os.path.join(datapath, 'Annotations', 'VID', phase, snippet_id)
  annotation_pattern = os.path.join(annotation_path, '*.xml')
  annotation_files = glob.glob(annotation_pattern)
  snippet_length = len(annotation_files)
  assert(snippet_length > 0)
  register = SnippetRegister()
  meta = parse_annotation_file(os.path.join(annotation_path, '000000.xml'))
  shots = Shot.from_anno_dicts(meta['object'], frame=0)
  register.register(shots)
  for i in range(1, snippet_length):
    anno = parse_annotation_file(os.path.join(annotation_path, '{:06d}.xml'.format(i)))
    shots = Shot.from_anno_dicts(anno['object'], frame=i)
    register.register(shots)
  snippets = register.close()
  print('{},{}'.format(len(list(meta['object'])), len(snippets)))
  return snippets

def create_tfrecords(snippet_list_file: str):
  body, _ = os.path.basename(snippet_list_file).split(sep='.')
  record_path = os.path.join(FLAGS.record_dir, '{}.tfrecords'.format(body))
  logging.info('W2:{}'.format(record_path))
  if os.path.exists(record_path):
    logging.info('PASS')
    # return
  writer = tf.python_io.TFRecordWriter(record_path)
  with open(snippet_list_file, 'r') as csvfile:
    for snippet_id, _ in csv.reader(csvfile, delimiter=' '):
      parse_snippet(datapath=FLAGS.data_dir, snippet_id=snippet_id, phase=FLAGS.phase)
      logging.info('\t{}'.format(snippet_id))
  writer.close()

def main(_):
  if not os.path.exists(FLAGS.data_dir):
    raise NotADirectoryError(FLAGS.data_dir)
  snippet_list_files = get_snippet_list_files(data_path=FLAGS.data_dir, phase=FLAGS.phase)
  for f in snippet_list_files[:1]:
    create_tfrecords(f)

  # sess = tf.Session()
  # sess.run(tf.global_variables_initializer())
  # sess.run(tf.local_variables_initializer())
  # coord = tf.train.Coordinator()
  # threads = tf.train.start_queue_runners(sess=sess, coord=coord)
  # # main
  # coord.request_stop()
  # coord.join(threads)

if __name__ == '__main__':
  tf.app.run()