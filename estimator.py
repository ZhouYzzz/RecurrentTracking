#!/usr/bin/python
import tensorflow as tf
import os, sys, glob
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from resnet_model import imagenet_resnet_v2

def main(_):
    network = imagenet_resnet_v2(resnet_size=50, num_classes=1000, data_format='channels_first')
    inputs_0 = tf.placeholder(tf.float32, shape=(None,224,224,3))
    inputs_1 = tf.placeholder(tf.float32, shape=(None,224,224,3))
    with tf.variable_scope('resnet'):
        outputs_0 = network(inputs_0, False)
    with tf.variable_scope('resnet', reuse=True):
        outputs_1 = network(inputs_1, False)
    for v in tf.global_variables():
        print v.name

if __name__ == '__main__':
    tf.app.run()

