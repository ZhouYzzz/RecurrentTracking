import tensorflow as tf


def input_fn():
  dataset = tf.data.TFRecordDataset(filenames=['/Users/zhouyz/Development/ILSVRC2015/train.tfrecords'])
  dataset = dataset.map(lambda s: tf.parse_single_sequence_example(s,
    context_features={
      'snippet_id': tf.FixedLenFeature([], tf.string),
      'snippet_length': tf.FixedLenFeature([], tf.int64)
    },
    sequence_features={
      'frames': tf.FixedLenSequenceFeature([1], tf.int64),
      'bndboxes': tf.FixedLenSequenceFeature([4], tf.int64)
    }))
  dataset.repeat
  iterator = dataset.make_one_shot_iterator()
  context, sequence = iterator.get_next()
  return context, sequence


def main(_):
  context, sequence = input_fn()
  with tf.Session() as sess:
    print(sess.run([context, sequence]))

if __name__ == '__main__':
  tf.app.run()