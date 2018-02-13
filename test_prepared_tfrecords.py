import tensorflow as tf
import glob

def input_fn():
  dataset = tf.data.TFRecordDataset(filenames=glob.glob('/tmp/*.tfrecords'))
  dataset = dataset.map(lambda s: tf.parse_single_sequence_example(s,
      context_features={
        'id': tf.FixedLenFeature(shape=[], dtype=tf.string)
      },
      sequence_features={
        'frame': tf.FixedLenSequenceFeature(shape=[1], dtype=tf.int64)
      }))
  dataset = dataset.batch(10)
  iterator = dataset.make_one_shot_iterator()
  context, sequence_example = iterator.get_next()
  features = sequence_example['frame']
  labels = features
  return features, labels

def model_fn(features, labels, mode, params):
  # print(features)
  # print(labels)
  inputs = tf.random_uniform(shape=(10, 32, 64, 64, 3))
  bndboxes = tf.random_uniform(shape=(10, 32, 4))
  outputs = track_model(inputs, bndboxes)
  loss = tf.reduce_mean(outputs)
  print(outputs)
  predictions = tf.cast(features, tf.float32)
  # loss = tf.constant(0.0)
  train_op = tf.assign_add(tf.train.get_or_create_global_step(), 1)
  train_op = tf.Print(train_op, [tf.train.get_or_create_global_step()])
  return tf.estimator.EstimatorSpec(
    mode=mode,
    predictions=predictions,
    loss=loss,
    train_op=train_op)

def track_model(frames, bndboxes):
  print(frames)
  inputs = tf.unstack(frames, axis=1)
  print(inputs)
  # cell = tf.nn.rnn_cell.BasicRNNCell(num_units=1024, activation=tf.nn.sigmoid, reuse=tf.AUTO_REUSE)
  cell_0 = tf.nn.rnn_cell.BasicLSTMCell(num_units=1024, reuse=tf.AUTO_REUSE)
  cell_1 = tf.nn.rnn_cell.BasicLSTMCell(num_units=4, reuse=tf.AUTO_REUSE)
  cell = tf.nn.rnn_cell.MultiRNNCell(cells=[cell_0, cell_1])
  state = cell.zero_state(batch_size=10, dtype=tf.float32)
  outputs = []
  for i in range(32):
    feature = cnn_model(inputs[i])
    output, state = cell(feature, state)
    outputs.append(output)
  # outputs, state = tf.nn.static_rnn(cell=cell, inputs=list(map(lambda i: tf.layers.flatten(i), inputs)), dtype=tf.float32)
  return outputs

def cnn_model(inputs):
  with tf.variable_scope('cnn', reuse=tf.AUTO_REUSE):
    inputs = tf.layers.conv2d(inputs, filters=64, kernel_size=(3, 3), strides=(2, 2))
    inputs = tf.layers.average_pooling2d(inputs, pool_size=(2, 2), strides=(2, 2))
    inputs = tf.layers.conv2d(inputs, filters=64, kernel_size=(3, 3), strides=(2, 2))
    inputs = tf.layers.average_pooling2d(inputs, pool_size=(2, 2), strides=(2, 2))
    # inputs = tf.layers.conv2d(inputs, filters=64, kernel_size=(3, 3), strides=(2, 2))
    # inputs = tf.layers.average_pooling2d(inputs, pool_size=(2, 2), strides=(2, 2))
    inputs = tf.layers.flatten(inputs)
  return inputs

# def track_cell(frame, bndbox):
  # return tf.nn.rnn_cell.BasicRNNCell
# class ImageLSTMCell(tf.nn.rnn_cell.BasicLSTMCell):
#   def __call__(self, inputs, state):
#     return None

def main():
  estimator = tf.estimator.Estimator(model_fn=model_fn)
  estimator.train(input_fn=input_fn, max_steps=10)
  # for item in estimator.predict(input_fn=input_fn):
  #   print(item)
  print('EOF')
  

if __name__ == '__main__':
  main()