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
  for id in tqdm(snippet_ids, desc='Total TFRecords: {}'.format(count)):
    count += create_tfrecords(os.path.join(dataset.annotations_dir, id))


if __name__ == '__main__':
  main()
