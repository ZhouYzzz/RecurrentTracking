DEF_PB_FILE = '/Users/zhouyz/Downloads/inception-2015-12-05/classify_image_graph_def.pb'
import tensorflow as tf

# graph = tf.import_graph_def(DEF_PB_FILE)
# tf.train.import_meta_graph(DEF_PB_FILE)
# with open(DEF_PB_FILE) as f:
  # tf.GraphDef(f.read())
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'


with tf.Session() as sess:
  with tf.gfile.FastGFile(DEF_PB_FILE, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
      tf.import_graph_def(graph_def, name='', return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,
              RESIZED_INPUT_TENSOR_NAME]))

print('EOP') # End Of Program
