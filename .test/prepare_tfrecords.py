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
  # print('{},{}'.format(len(list(meta['object'])), len(snippets)))
  return meta, snippets


# def parse_annotation_tree(tree):
#     return {'trackid': _trackid.text,
#             'name': _name.text,
#             'bndbox': [int(e.text) for e in _bndbox[0:4]], # xmax, xmin, ymax, ymin
#             'occluded': int(_occluded.text),
#             'generated': int(_generated.text)}
#   return {'folder': _folder.text,
#           'filename': _filename.text,
#           'size': {'width': int(_size[0].text), 'height': int(_size[1].text)},
#           'object': list(map(parse_object, _object))}
def snippet2example(meta, s: Snippet):
  context = tf.train.Features(feature={
    'id': feature_bytes(meta['folder']),
    'width': feature_int64(meta['size']['width']),
    'height': feature_int64(meta['size']['height']),
    'length': feature_int64(32)
  })
  feature_lists = tf.train.FeatureLists(feature_list={
    'frame': feature_list_int64(s.entities['frame']),
    'bndbox': feature_list_int64(s.entities['bndbox'])
  })
  sequence_example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)
  return sequence_example

def split_snippets(s: Snippet, l: int):
  length = s.length
  n = max(length // l, 1) # at least 1
  return s.split(n, l)

def create_tfrecords(snippet_list_file: str):
  body, _ = os.path.basename(snippet_list_file).split(sep='.')
  record_path = os.path.join(FLAGS.record_dir, '{}.tfrecords'.format(body))
  logging.info('W2:{}'.format(record_path))
  if os.path.exists(record_path):
    logging.info('PASS')
    return
  writer = tf.python_io.TFRecordWriter(record_path)
  with open(snippet_list_file, 'r') as csvfile:
    for snippet_id, _ in csv.reader(csvfile, delimiter=' '):
      # for each snippet id, get all the snippets it contains
      meta, snippets = parse_snippet(datapath=FLAGS.data_dir, snippet_id=snippet_id, phase=FLAGS.phase)
      # split each snippet to fixed length subsnippets
      subsnippets = [] # type: List[Snippet]
      for s in snippets:
        subsnippets += split_snippets(s, 32)
      examples = list(map(lambda s: snippet2example(meta, s), subsnippets))
      for e in examples:
        writer.write(e.SerializeToString())
      logging.info('\t{}, {}'.format(snippet_id, len(subsnippets)))
  writer.close()

def main(_):
  if not os.path.exists(FLAGS.data_dir):
    raise NotADirectoryError(FLAGS.data_dir)
  snippet_list_files = get_snippet_list_files(data_path=FLAGS.data_dir, phase=FLAGS.phase)
  for f in snippet_list_files[:1]:
    # for each snippet list file, e.g. train_01.txt
    # create corresponding tfrecords file train_01.tfrecords
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