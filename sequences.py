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
        tf.train.match_filenames_once(sequence_list_filename_pattern))
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
    annotations_folder = os.path.join(FLAG.data_path,'Annotations','VID',sequence_np)
    images_folder = os.path.join(FLAGS.data_path,'Data','VID',sequence_np)
    first_frame_annotation_file = os.path.join(annotations_folder, '000000.xml')
    tree = ET.parse(first_frame_annotation_file)
    size_node = tree.getroot().find('size')
    # size = [width, height]
    size = np.array([np.float32(x.text) for x in size_node.getchildren()], np.float32)
    objects = tree.getroot().findall('object')
    # bndboxs = [n] rows of [xmax,xmin,ymax,ymin]
    bndboxs = np.array([[float(x.text) for x in obj.find('bndbox').getchildren()] for obj in objects],np.float32)
    num_objs = np.array(len(objects), np.float32)
    return [size, num_objs, bndboxs]
  with tf.name_scope('sequence_info'):
    return tf.py_func(_py_load_sequence_info, [sequence], [tf.float32])

def main(_):
  sequence = load_sequence_list()
  info = load_sequence_info(sequence)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(200):
      print sess.run(info[1])
    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
  tf.app.run()
