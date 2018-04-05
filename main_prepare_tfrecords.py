"""Create TFRecords files from ILSVRC2015"""
import tensorflow as tf
import tempfile, os, argparse
from multiprocessing import Pool
from tqdm import tqdm

from ilsvrc2015 import ILSVRC2015, PHASE
from annotations import parse_annotation_folder


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset_dir', default='/home/zhouyz/ILSVRC2015/', type=str, help='ILSVRC2015 root directory')
parser.add_argument('--output_dir', default=tempfile.mkdtemp(), type=str)
parser.add_argument('--records_prefix', default='ilsvrc2015.', type=str)


FLAGS, _ = parser.parse_known_args()


def create_tfrecords(annotation_folder):
  writer = tf.python_io.TFRecordWriter(
    path=tempfile.mktemp(suffix='.tfrecords', prefix=FLAGS.records_prefix, dir=FLAGS.output_dir))
  streams = parse_annotation_folder(annotation_folder)
  for s in streams:
    writer.write(s.serializeToTFSequenceExample().SerializeToString())
  writer.close()
  return len(streams)


def create_fixed_lengthed_tfrecords(annotation_folder, length=32):
  writer = tf.python_io.TFRecordWriter(
    path=tempfile.mktemp(suffix='.tfrecords', prefix=FLAGS.records_prefix, dir=FLAGS.output_dir))
  streams = parse_annotation_folder(annotation_folder)
  splitted_streams = []
  for s in streams:
    splitted_streams += s.splitIntoStreams(n=s.length//length + 1, l=length)
  for s in splitted_streams:
    writer.write(s.serializeToTFSequenceExample().SerializeToString())
  writer.close()
  return len(splitted_streams)


def main():
  print('FLAGS:', FLAGS)
  dataset = ILSVRC2015(FLAGS.dataset_dir)
  snippet_ids = dataset.GetSnippetIDs(phase=PHASE.TRAIN)
  ## Using multiprocessing
  # with Pool(8) as p:
  #   r = list(tqdm(
  #     p.imap(create_tfrecords, map(lambda i: os.path.join(dataset.annotations_dir, i), snippet_ids)),
  #     total=len(snippet_ids)
  #   ))
  count = 0
  t = tqdm(snippet_ids)
  for id in t:
    count += create_fixed_lengthed_tfrecords(os.path.join(dataset.annotations_dir, id))
    t.set_description(desc='Total records {}'.format(count))


if __name__ == '__main__':
  main()
