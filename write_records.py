import tensorflow as tf
from annotations import parse_snippet_annotations
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

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
def _bytes_feature(value):
  return tf.train.Feature(
    bytes_list=tf.train.BytesList(value=[str(value)]))
def _float32_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))
def _int64_feature_list(values):
  return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])
def _bytes_feature_list(values):
  return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])
def _float32_feature_list(values):
  return tf.train.FeatureList(feature=[_float32_feature(v) for v in values])


def _snippet_list_files(data_path, phase='train'):
  if phase not in PHASES:
    raise ValueError('`phase` should be one of {}, got \'{}\''.format(PHASES, phase))
  pattern = os.path.join(data_path, 'ImageSets', TASK, '{}*.txt'.format(phase))
  return glob.glob(pattern)


def _parse_snippet(data_path, snippet_id, phase='train'):
  logging.info('parse snippet {}'.format(snippet_id))
  snippet_anno_dir = os.path.join(data_path, 'Annotations', TASK, phase, snippet_id)
  results = parse_snippet_annotations(snippet_anno_dir=snippet_anno_dir)
  serialized_examples = []
  for s in results['subsnippets']:
    context = tf.train.Features(feature={
      'snippet_id': _bytes_feature(results['snippet_id']),
      'snippet_length': _int64_feature([s['length']]),
    })
    feature_lists = tf.train.FeatureLists(feature_list={
      'frames': _int64_feature_list(s['frames']),
      'bndboxes': _int64_feature_list(s['bndboxes'])
    })
    example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)
    serialized_examples.append(example.SerializeToString())
  # snippet_img_dir = os.path.join(data_path, 'Data', TASK, phase, snippet_id)
  return serialized_examples


def main(_):
  assert os.path.exists(FLAGS.data_path)
  snippet_list_files = _snippet_list_files(FLAGS.data_path, phase='train')
  snippet_list_files = snippet_list_files[0:1]
  writer = tf.python_io.TFRecordWriter(FLAGS.record_file)
  for snippet_list_file in snippet_list_files:
    with open(snippet_list_file, 'r') as csvfile:
      for [snippet_id, _] in csv.reader(csvfile, delimiter=' '):
        examples = _parse_snippet(data_path=FLAGS.data_path, snippet_id=snippet_id, phase='train')
        [writer.write(ex) for ex in examples]
  writer.close()


if __name__ == '__main__':
  # tf.logging.set_verbosity(tf.logging.INFO)
  # logging.Logger(name='LOGGER', level=logging.INFO)
  logging.basicConfig(level=logging.INFO)
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
