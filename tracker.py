import tensorflow as tf
from input_fn import fixed_length_input_fn
from ilsvrc2015 import ILSVRC2015
from utils.search_window import get_search_window_for_batch
from resnet.resnet_model import block_layer, imagenet_resnet_v2, conv2d_fixed_padding, batch_norm_relu, building_block, bottleneck_block


def imagenet_resnet_v2_generator(block_fn, layers, num_classes,
                                 data_format=None):
  """Generator for ImageNet ResNet v2 models.

  Args:
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    layers: A length-4 array denoting the number of blocks to include in each
      layer. Each layer consists of blocks that take inputs of the same size.
    num_classes: The number of possible classes for image classification.
    data_format: The input format ('channels_last', 'channels_first', or None).
      If set to None, the format is dependent on whether a GPU is available.

  Returns:
    The model function that takes in `inputs` and `is_training` and
    returns the output tensor of the ResNet model.
  """
  if data_format is None:
    data_format = (
        'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

  def model(inputs, is_training):
    """Constructs the ResNet model given the inputs."""
    if data_format == 'channels_first':
      # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
      # This provides a large performance boost on GPU. See
      # https://www.tensorflow.org/performance/performance_guide#data_formats
      inputs = tf.transpose(inputs, [0, 3, 1, 2])

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=64, kernel_size=7, strides=2,
        data_format=data_format)
    inputs = tf.identity(inputs, 'initial_conv')
    inputs = tf.layers.max_pooling2d(
        inputs=inputs, pool_size=3, strides=2, padding='SAME',
        data_format=data_format)
    inputs = tf.identity(inputs, 'initial_max_pool')

    inputs = block_layer(
        inputs=inputs, filters=64, block_fn=block_fn, blocks=layers[0],
        strides=1, is_training=is_training, name='block_layer1',
        data_format=data_format)
    block_layer1 = tf.layers.flatten(inputs)
    inputs = block_layer(
        inputs=inputs, filters=128, block_fn=block_fn, blocks=layers[1],
        strides=2, is_training=is_training, name='block_layer2',
        data_format=data_format)
    block_layer2 = tf.layers.flatten(inputs)
    inputs = block_layer(
        inputs=inputs, filters=256, block_fn=block_fn, blocks=layers[2],
        strides=2, is_training=is_training, name='block_layer3',
        data_format=data_format)
    block_layer3 = tf.layers.flatten(inputs)
    inputs = block_layer(
        inputs=inputs, filters=512, block_fn=block_fn, blocks=layers[3],
        strides=2, is_training=is_training, name='block_layer4',
        data_format=data_format)
    block_layer4 = tf.layers.flatten(inputs)
    # inputs = batch_norm_relu(inputs, is_training, data_format)
    # inputs = tf.layers.average_pooling2d(
    #     inputs=inputs, pool_size=7, strides=1, padding='VALID',
    #     data_format=data_format)
    # inputs = tf.identity(inputs, 'final_avg_pool')
    # inputs = tf.reshape(inputs,
    #                     [-1, 512 if block_fn is building_block else 2048])
    # inputs = tf.layers.dense(inputs=inputs, units=num_classes)
    # inputs = tf.identity(inputs, 'final_dense')
    inputs = tf.concat([block_layer3, block_layer4], axis=1)
    return inputs

  return model


# def resnet_feature_extractor()


class TrackerCell(tf.nn.rnn_cell.MultiRNNCell):
  def __init__(self, cells=[]):
    self._batch_size = 1
    model = imagenet_resnet_v2_generator(bottleneck_block, [3, 4, 6, 3], None, data_format='channels_first')
    self._feature_extract_fn = lambda inputs: model(inputs, is_training=True)
    return super(TrackerCell, self).__init__(cells=cells, state_is_tuple=True)

  def call(self, inputs, state):
    # inputs should be a touple of (image, bndbox)
    print(inputs)
    (i0, i1), (b0, b1) = inputs
    search_window = get_search_window_for_batch(b0)
    CROP_SIZE = (128, 128)
    ic0 = tf.image.crop_and_resize(image=i0,
                                   boxes=search_window,
                                   box_ind=tf.range(0, self._batch_size),
                                   crop_size=CROP_SIZE)
    ic1 = tf.image.crop_and_resize(image=i1,
                                   boxes=search_window,
                                   box_ind=tf.range(0, self._batch_size),
                                   crop_size=CROP_SIZE)
    # with tf.variable_scope('FeatureExt', reuse=tf.AUTO_REUSE):
    #   oc0 = self._feature_extract_fn(ic0)
    with tf.variable_scope('FeatureExt', reuse=tf.AUTO_REUSE):
      oc1 = self._feature_extract_fn(ic1)
    # with tf.variable_scope('FC', reuse=tf.AUTO_REUSE):
    #   oc0 = tf.layers.dense(oc0, 2048, activation=tf.nn.leaky_relu)
    with tf.variable_scope('FC', reuse=tf.AUTO_REUSE):
      oc1 = tf.layers.dense(oc1, 2048, activation=tf.nn.leaky_relu)
    # print(i0, i1, b0, b1, oc0, oc1)
    # inputs = tf.layers.flatten(i0)
    return super(TrackerCell, self).call(inputs=oc1, state=state)


def paired_sequence(seq):
  return [(x0, x1) for x0, x1 in zip(seq[:-1], seq[1:])]


def supervised_tracker(images,
                       bndboxes,
                       time_len: int = 32,
                       batch_size: int = 1,
                       crop_size: (int, int) = (224, 224)):
  seq_inputs = tf.unstack(images, num=time_len, axis=1)
  seq_bndboxes = tf.unstack(bndboxes, num=time_len, axis=1)
  seq_p_inputs = paired_sequence(seq_inputs)
  seq_p_bndboxes = paired_sequence(seq_bndboxes)

  # search_window = get_search_window_for_batch(seq_bndboxes[0])
  # cropped = tf.image.crop_and_resize(image=seq_inputs[0],
  #                                    boxes=search_window,
  #                                    box_ind=tf.range(0, batch_size),
  #                                    crop_size=crop_size)
  # feature_extractor = imagenet_resnet_v2(50, num_classes=1001, data_format='channels_first')
  # graph = tf.get_default_graph()
  # with tf.variable_scope('ResNet50', reuse=tf.AUTO_REUSE):
  #   outputs = feature_extractor(cropped, is_training=True)
  #   block1 = graph.get_tensor_by_name('ResNet50/block_layer1:0')
  #   block3 = graph.get_tensor_by_name('ResNet50/block_layer3:0')
  #   block4 = graph.get_tensor_by_name('ResNet50/block_layer4:0')
  #   print(block1, block3, block4)
  #
  # MODEL_DIR = '../resnet50_2017_11_30'
  # tf.train.init_from_checkpoint(MODEL_DIR, assignment_map={'/': 'ResNet50/'})
  # tf.nn.static_rnn()


  cell = TrackerCell(cells=[tf.nn.rnn_cell.LSTMCell(1024),
                            tf.nn.rnn_cell.LSTMCell(4)])
  # zero_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)
  # cell.call(inputs=(seq_p_inputs[0], seq_p_bndboxes[0]), state=zero_state)
  outputs, state = tf.nn.static_rnn(cell=cell,
                                    inputs=list(zip(seq_p_inputs, seq_p_bndboxes)),
                                    initial_state=cell.zero_state(batch_size, tf.float32))
  tf.train.init_from_checkpoint('../resnet50_2017_11_30', assignment_map={'/': 'FeatureExt/'})

  return None, None, None, None
  # return search_window, seq_bndboxes[0], cropped, outputs
  # rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=10)
  # zero_state = rnn_cell.zero_state(batch_size=2, dtype=tf.float32)
  # tf.nn.static_rnn(rnn_cell,
  #                  inputs=seq_inputs,
  #                  initial_state=zero_state)


def deployed_tracker(images,
                     init_bndbox):
  pass


if __name__ == '__main__':
  dataset = ILSVRC2015(root_dir='/home/zhouyz/ILSVRC2015')
  context, sequence_example = fixed_length_input_fn(ilsvrc=dataset, batch_size=1)
  print(context)
  print(sequence_example)
  r = supervised_tracker(images=sequence_example['images'],
                         bndboxes=sequence_example['bndboxes'],
                         batch_size=1)
  print(r)
  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())
  # print(r[3].eval())
  # tf.image.crop_and_resize()
  # with tf.Session() as sess:
  #   for _ in range(10):
  #     sess.run(r)
  # tf.nn.static_rnn()
