"""Create Input Function of ILSVRC2015 used for TF.Estimator"""
import tensorflow as tf
import glob, os
from ilsvrc2015 import ILSVRC2015


def preprocess(context, sequence_example, dataset_data_dir: str):
  def _parse_image(folder_t, frame_t, size_t):
    image_filename = tf.string_join(
                      [dataset_data_dir,
                       folder_t,
                       tf.string_join([tf.as_string(frame_t[0], width=6, fill='0'), '.JPEG'])],
                      separator='/')
    image_raw = tf.read_file(image_filename)
    image_decoded = tf.image.decode_jpeg(image_raw, channels=3)
    image_decoded = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
    image_resized = tf.image.resize_images(image_decoded, [360, 480])
    # image_resized = image_decoded
    return image_resized

  def _parse_bndbox(bndboxes_t, size_t):  # size_t: [width, height]
    bndboxes_t = tf.cast(bndboxes_t, dtype=tf.float32)
    size_t = tf.cast(size_t, dtype=tf.float32)
    bndboxes_t = tf.div(bndboxes_t, [size_t[0], size_t[0], size_t[1], size_t[1]])
    return bndboxes_t

  sequence_example['images'] = tf.map_fn(
    lambda frame: _parse_image(context['folder'], frame, context['size']),
    sequence_example['frames'],
    dtype=tf.float32
  )
  sequence_example['bndboxes'] = tf.map_fn(
    lambda bndboxes: _parse_bndbox(bndboxes, context['size']),
    sequence_example['bndboxes'],
    dtype=tf.float32
  )
  return context, sequence_example


def fixed_length_input_fn(phase: str = 'train',
                          batch_size: int = 16,
                          num_epoches: int = None,
                          ilsvrc: ILSVRC2015 = None):
  tfrecord_files = glob.glob(os.path.join(ilsvrc.tfrecords_dir, 'fixed_length', phase, '*.tfrecords'))
  if len(tfrecord_files) == 0: raise FileNotFoundError('No existing tfrecords. Run generating script first.')
  dataset = tf.data.TFRecordDataset(tfrecord_files)  # type: tf.data.Dataset
  dataset = dataset.map(
    lambda s: tf.parse_single_sequence_example(s,
                                               context_features={
                                                 'folder': tf.FixedLenFeature([], tf.string),
                                                 'size': tf.FixedLenFeature([2], tf.int64),
                                                 'length': tf.FixedLenFeature([], tf.int64)
                                               },
                                               sequence_features={
                                                 'frames': tf.FixedLenSequenceFeature([1], tf.int64),
                                                 'bndboxes': tf.FixedLenSequenceFeature([4], tf.int64)
                                               }))
  dataset = dataset.map(
    lambda c, s: preprocess(context=c, sequence_example=s, dataset_data_dir=os.path.join(ilsvrc.data_dir, phase)))
  dataset = dataset.batch(batch_size=batch_size)
  dataset = dataset.repeat(count=num_epoches)
  iterator = dataset.make_one_shot_iterator()
  context, sequence = iterator.get_next()
  return context, sequence


if __name__ == '__main__':
  dataset = ILSVRC2015(root_dir='/home/zhouyz/ILSVRC2015')
  e = fixed_length_input_fn(ilsvrc=dataset, batch_size=16)
  print(e)
  sess = tf.Session()
  import time
  t = time.time()
  for _ in range(10):
    c, s = sess.run(e)  # around 2 seconds per iter
    print(time.time() - t)
  # print(s['images'].shape)
