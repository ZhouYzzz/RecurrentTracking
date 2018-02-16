#!/usr/bin/python
import tensorflow as tf 
print 'Using Tensorflow', tf.__version__
import numpy as np
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from glob import glob
import xml.etree.ElementTree as ET
from parsexml import parsexml

flags = tf.flags
flags.DEFINE_string("data_path", '/home/spark/data/ILSVRC2015/ILSVRC2015',
        "Where the training/test/val data is stored.")
FLAGS = flags.FLAGS

def input_parser(sequence_folder, object_identifier, selected_frame_list):
    return sequence_folder, object_identifier

def get_all_sequence_identifiers(dataset_identifier='train'):
    sequence_list_filename_pattern = os.path.join(
            FLAGS.data_path,'ImageSets','VID',dataset_identifier+'_1.txt')
    sequence_list_filenames = glob(sequence_list_filename_pattern)
    sequence_list_filenames.sort()
    sequence_identifier_list = list()
    for fname in sequence_list_filenames:
        sequence_identifier_list += np.loadtxt(fname,dtype=str)[:,0].tolist()
    return map(lambda x: dataset_identifier+'/'+x ,sequence_identifier_list)

def get_sequence_info(sequence):
    pass

def parse_anno(filename):
    tree = ET.parse(filename)
    #size_node = tree.getroot().find('size')
    #size = [int(x.text) for x in size_node.getchildren()]
    #objects = tree.getroot().findall('object')
    #num_object = len(objects)
    objs = tree.findall('object')
    num_object = len(objs)
    return num_object

def construct_subsequences(sequence_identifier):
    sequence_anno_filename_pattern = os.path.join(
            FLAGS.data_path,'Annotations','VID',sequence_identifier,'*.xml')
    sequence_frame_filename_pattern = os.path.join(
            FLAGS.data_path,'Data','VID',sequence_identifier,'*.JPEG')
    sequence_anno_filenames = sorted(glob(sequence_anno_filename_pattern))
    sequence_frame_filenames = sorted(glob(sequence_frame_filename_pattern))
    assert(len(sequence_anno_filenames)==len(sequence_frame_filenames))
    num_frame = len(sequence_anno_filenames)
    anno = parsexml(sequence_anno_filenames[0])
    num_obj = len(anno.objs)
    for filename in sequence_anno_filenames[1:]:
        print '\t', filename
        assert(len(parsexml(filename).objs)==num_obj)
    return anno

def dataset_components(sequence_identifiers):
    # generate several tensors required for dataset construction
    for identifier in sequence_identifiers:
        anno = construct_subsequences(identifier)
        print identifier, len(anno.objs)

def main(_):
    sequence_identifiers = get_all_sequence_identifiers()
    dataset_components(sequence_identifiers)
    sequence_folders = tf.placeholder(tf.string)
    object_identidiers = tf.placeholder(tf.int64)
    selected_frame_lists = tf.placeholder(tf.int64)
    dataset = tf.data.Dataset.from_tensor_slices((sequence_folders, object_identidiers, selected_frame_lists))
    dataset = dataset.map(input_parser)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    with tf.Session() as sess:
        sess.run(iterator.initializer, feed_dict={
            sequence_folders:['/path/to'],
            object_identidiers:[0],
            selected_frame_lists:[1]})
        value = sess.run(next_element)
        print value

if __name__ == '__main__':
    tf.app.run()

