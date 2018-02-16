#!/usr/bin/python
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
from glob import glob
from xml.etree import ElementTree as ET

flags = tf.flags
flags.DEFINE_string('data_path', '/home/spark/data/ILSVRC2015/ILSVRC2015',
        'Where the train/test/val data stores')
flags.DEFINE_string('record_file', '/tmp/test.tfrecords',
        'The record file storing track examples')
FLAGS = flags.FLAGS

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
def _bytes_feature(value):
    return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[str(value)]))
def _int64_feature_list(values):
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])
def _bytes_feature_list(values):
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])

def load_sequence(dataset_identifier='train'):
  with tf.name_scope('sequence_list'):
    sequence_list_filename_pattern = os.path.join(
            FLAGS.data_path,'ImageSets','VID',dataset_identifier+'*.txt');
    sequence_list_filename_queue = tf.train.string_input_producer(
            tf.train.match_filenames_once(sequence_list_filename_pattern),
            shuffle=False,
            num_epochs=1)
    # parse sequence_list_files to sequence_list
    reader = tf.TextLineReader()
    _, value = reader.read(sequence_list_filename_queue)
    record_defaults = [['sequence'],[1]]
    sequence, _ = tf.decode_csv(
            value, record_defaults=record_defaults, field_delim=' ')
    sequence = tf.string_join([dataset_identifier, sequence], separator='/')
    sequence_list = tf.FIFOQueue(capacity=32, dtypes=tf.string)
    tf.train.add_queue_runner(tf.train.QueueRunner(
        sequence_list, [sequence_list.enqueue(sequence)]))
    # return the dequeue handle (the sequence)
    return sequence_list.dequeue()

def _parse_object(node):
    trackid = int(node.find('trackid').text)
    bndbox = [int(n.text) for n in node.find('bndbox').getchildren()]
    occluded = [int(node.find('occluded').text)]
    generated = [int(node.find('generated').text)]
    return trackid, bndbox, occluded, generated

class SequenceParser:
    def __init__(self, sequence):
        self._sequence = sequence
        self._anno_folder = os.path.join(
                FLAGS.data_path,'Annotations','VID',self._sequence)
        self._data_folder = os.path.join(
                FLAGS.data_path,'Data','VID',self._sequence)
        self._anno_list = glob(os.path.join(self._anno_folder,'*.xml'))
        self._data_list = glob(os.path.join(self._data_folder,'*.JPEG'))
        self._total_length = len(self._anno_list)
        assert(tf.gfile.Exists(self._anno_folder))
        assert(self._total_length == len(self._data_list))
        # a helper structure for object identifying
        self._dicts = [dict() for _ in xrange(4)] # store [frame, box, o, g]
        self._trackids = list()

    def _register(self, trackid, *items):
        assert(len(items)==4)
        if trackid not in self._trackids:
            self._trackids.append(trackid)
        for i in xrange(4):
            if trackid not in self._dicts[i]:
                self._dicts[i][trackid] = list() # init an empty list
            self._dicts[i][trackid].append(items[i])

    def parse(self):
        for frame in xrange(self._total_length):
            anno_file = os.path.join(self._anno_folder,'%06d.xml'%(frame))
            tree = ET.parse(anno_file)
            objects = tree.findall('object')
            for o in objects:
                trackid, bndbox, occluded, generated = _parse_object(o)
                self._register(trackid, [frame], bndbox, occluded, generated)
        return self._construct_sequence_examples()

    def _construct_sequence_examples(self):
        num_tracks = len(self._trackids)
        sequence_examples = list()
        for trackid in self._trackids:
            context = tf.train.Features(feature={
                'sequence': _bytes_feature(self._sequence),
                'length': _int64_feature([len(self._dicts[0][trackid])])
                })
            feature_lists = tf.train.FeatureLists(feature_list={
                'frame': _int64_feature_list(self._dicts[0][trackid]),
                'bndbox': _int64_feature_list(self._dicts[1][trackid]),
                'occluded': _int64_feature_list(self._dicts[2][trackid]),
                'generated': _int64_feature_list(self._dicts[3][trackid])
                })
            sequence_examples.append(
                    tf.train.SequenceExample(context=context,
                        feature_lists = feature_lists).SerializeToString())
        return sequence_examples

def construct_tracks_from_sequence(sequence):
    def _construct_tracks(sequence):
        sequence_examples = SequenceParser(sequence).parse()
        return np.array(sequence_examples)
    #sequence = tf.Print(sequence,[sequence])
    tracks_queue = tf.FIFOQueue(capacity=32, dtypes=tf.string)
    tracks = tf.py_func(_construct_tracks,[sequence],[tf.string])
    enqueue = tracks_queue.enqueue_many(tracks)
    tf.train.add_queue_runner(tf.train.QueueRunner(
        tracks_queue, [enqueue]))
    return tracks_queue.dequeue()


def main(_):
    # WRITE PHASE
    print 'WRITE PHASE'
    record_writer = tf.python_io.TFRecordWriter(FLAGS.record_file)
    sequence = load_sequence()
    track = construct_tracks_from_sequence(sequence)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in xrange(30):
            value = sess.run(track)
            record_writer.write(value)
        coord.request_stop()
        coord.join(threads)
    record_writer.close()

    # READ PHASE
    print 'READ PHASE'
    record_reader = tf.TFRecordReader()
    record_queue = tf.train.string_input_producer([FLAGS.record_file])
    key, value = record_reader.read(record_queue)
    context, sequence_example = tf.parse_single_sequence_example(
            value,
            context_features={
                'sequence': tf.FixedLenFeature([], tf.string),
                'length': tf.FixedLenFeature([], tf.int64)
            },
            sequence_features={
                'frame': tf.FixedLenSequenceFeature([], tf.int64),
                #'bndbox': tf.FixedLenSequenceFeature([4], tf.int64),
                #'occluded': tf.FixedLenSequenceFeature([], tf.int64),
                #'generated': tf.FixedLenSequenceFeature([], tf.int64)
            })
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in xrange(30):
            print sess.run([context, sequence_example])
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    tf.app.run()

