#!/usr/bin/python

import tensorflow as tf
import numpy as np
import xml.etree.ElementTree as ET
import os

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("data_path", '/home/spark/data/ILSVRC2015/ILSVRC2015',
                    "Where the training/test/val data is stored.")
FLAGS = flags.FLAGS

def load_sequence_list(dataset_identifier='train'):
  with tf.name_scope('sequence_list'):
    sequence_list_filename_pattern = os.path.join(
        FLAGS.data_path,'ImageSets','VID',dataset_identifier+'*.txt');
    # TODO: edit this string_queue to specify epoches
    sequence_list_filename_queue = tf.train.string_input_producer(
        tf.train.match_filenames_once(sequence_list_filename_pattern), shuffle=False)
    # parse sequence_list_files to sequence_list
    reader = tf.TextLineReader()
    _, value = reader.read(sequence_list_filename_queue)
    record_defaults = [['sequence'],[1]]
    sequence, _ = tf.decode_csv(
        value, record_defaults=record_defaults, field_delim=' ')
    sequence = tf.string_join([dataset_identifier, sequence], separator='/')
    sequence_list = tf.FIFOQueue(capacity=100, dtypes=tf.string)
    tf.train.add_queue_runner(tf.train.QueueRunner(
      sequence_list, [sequence_list.enqueue(sequence)]))
    # return the dequeue handle (the sequence)
    return sequence_list.dequeue()


def load_sequence_info(sequence):
  def _py_load_sequence_info(sequence_np):
    # a numpy wrapper for loading sequence_info
    annotations_folder = os.path.join(FLAGS.data_path,'Annotations','VID',sequence_np)
    images_folder = os.path.join(FLAGS.data_path,'Data','VID',sequence_np)
    first_frame_annotation_file = os.path.join(annotations_folder, '000000.xml')
    tree = ET.parse(first_frame_annotation_file)
    size_node = tree.getroot().find('size')
    # size = [width, height]
    size = np.array([np.int32(x.text) for x in size_node.getchildren()], np.int32)
    objects = tree.getroot().findall('object')
    # bndboxs = [n] rows of [xmax,xmin,ymax,ymin]
    # bndboxs = np.array([[float(x.text) for x in obj.find('bndbox').getchildren()] for obj in objects],np.float32)
    num_objs = np.array(len(objects), np.int32)
    # sequence_np = np.array([sequence_np]*len(objects))
    return [size, num_objs] # [(0) size of image, (1) num of objects]
  with tf.name_scope('sequence_info'):
    return [sequence] + tf.py_func(_py_load_sequence_info, [sequence], [tf.int32, tf.int32])

#def filter_sequence(sequence, info):
#    def _py_filter_sequence_info(sequence_np, num_objs_np):
#      sequence_np = np.array([sequence_np]*num_objs_np[0])
#      index_obj_np = np.range(num_objs_np[0], np.int32)
#      return [sequence_np, index_obj_np]
#    return tf.py_func(_py_filter_sequence_info, [sequence, info[1]], [tf.string, tf.int32])

def main(_):
  sequence = load_sequence_list()
  sequence, size, num_objs = load_sequence_info(sequence)
  #sequence, ind_obj = filter_sequence(sequence, info)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(100):
      print i, sess.run(sequence)
    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
  tf.app.run()
