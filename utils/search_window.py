import tensorflow as tf


def get_search_window_from_bndbox(bndbox, scale=2.):  # bndbox: (4)
  """Transform anno bndbox to search box(used in crop_and_resize) with a certain scale factor"""
  x_c = (bndbox[0] + bndbox[1]) / 2
  y_c = (bndbox[2] + bndbox[3]) / 2
  w = (bndbox[1] - bndbox[0])
  h = (bndbox[3] - bndbox[2])
  bndbox = tf.stack([y_c - h * scale / 2.,
                     x_c - w * scale / 2.,
                     y_c + h * scale / 2.,
                     x_c + w * scale / 2.])
  return bndbox


def get_search_window_for_batch(batch_of_bndboxes, scale=2.):  # bndbox: (None, 4)
  return tf.map_fn(lambda b: get_search_window_from_bndbox(b, scale=scale), batch_of_bndboxes)
