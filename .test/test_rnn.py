import tensorflow as tf
import numpy as np

num_steps = 6
batch_size = 10

def build_graph(x):
  inputs = [tf.identity(x) for _ in range(num_steps)]
  cell = tf.nn.rnn_cell.BasicRNNCell(num_units=10, activation=tf.nn.relu, reuse=tf.AUTO_REUSE)
  state = cell.zero_state(dtype=tf.float32, batch_size=batch_size)
  outputs = []
  for inp in inputs:
    output, state = cell(inp, state)
    outputs.append(output)
  return outputs

class PreprocessedCell(tf.nn.rnn_cell.BasicRNNCell):
  def call(self, inputs, state):
    inputs = tf.multiply(inputs, inputs)
    return super(PreprocessedCell, self).call(inputs, state)


class TrackCell(tf.nn.rnn_cell.MultiRNNCell):
  def __init__(self, feature_model, cells, state_is_tuple=True):
    self._feature_model = feature_model
    super(TrackCell, self).__init__(cells, state_is_tuple=state_is_tuple)

  def call(self, inputs, state):
    # preprocess
    inputs = self._feature_model(inputs)
    # excute RNNs
    outputs, state = super(TrackCell, self).call(inputs, state)
    # postprocess
    return outputs, state


def search_window(frame, bndbox):
  """
  Calculate search window given bndbox, which is a 4D vector
  """
  # return oh, ow, th, tw
  # tf.image.crop_to_bounding_box(image, oh, ow, th, tw)
  pass


def build_graph_v2(x):
  inputs = [tf.identity(x) for _ in range(num_steps)]
  cell = PreprocessedCell(num_units=10, activation=tf.nn.relu, reuse=tf.AUTO_REUSE)
  state = cell.zero_state(dtype=tf.float32, batch_size=batch_size)
  outputs = []
  for inp in inputs:
    output, state = cell(inp, state)
    outputs.append(output)
  return outputs

def main():
  x = tf.placeholder(tf.float32, shape=(batch_size, 4))
  y = build_graph_v2(x)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    rand_array = np.random.rand(10, 4)
    print(sess.run(y, feed_dict={x: rand_array}))

if __name__ == '__main__':
  main()
  print("EOP")
