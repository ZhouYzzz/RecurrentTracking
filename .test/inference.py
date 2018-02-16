from typing import List, Callable, Union, Any
import tensorflow as tf

class Foo(object):
  """
  Here is the doc for class `Foo`
  1. aaa
  2. bbb

  *it*, **bold**
  ## comment
  ```
  code example is here:
  if a = b
    else
      end
  ```
  """
  def __init__(self):
    """
    Here is the doc for method `__init__`
    """
    pass

  def bar(self):
    pass

def create_foo():
  return Foo()


def afun(x: List[str]) -> str:
  return x


def bfun(x: Any):
  return isinstance(x, str)


def feature_bytes(value: Union[str, bytes]):
  if isinstance(value, str):
    value = bytes(value, 'utf-8')
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def main(_): 
  bfun('aaa')
  print(tf.train.BytesList(value=[b'abc',b'def']))
  # print()
  feature = feature_bytes(value='asdashfghfhgfhg')
  bfeature = feature_bytes(value=b'thisisbtyoes')
  example = tf.train.Example(features=tf.train.Features(feature={'name': feature, 'bytes': bfeature}))
  print(example.SerializeToString())
  

if __name__ == '__main__':
  main(None)