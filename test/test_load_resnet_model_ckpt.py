import tensorflow as tf
from resnet.resnet_model import imagenet_resnet_v2


# the model dir contains a pre-trained weights file, saved as a checkpoint file
MODEL_DIR = '../resnet50_2017_11_30'


def main(_):
  inputs = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
  model = imagenet_resnet_v2(50, 1001, data_format='channels_first')
  outputs = model(inputs, is_training=False)
  tf.train.init_from_checkpoint(MODEL_DIR, assignment_map={'/': '/'})

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
