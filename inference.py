
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

def main(_):
  foo = create_foo()
  foo.bar()
  import os
  os.path.join('')
  

if __name__ == '__main__':
  main(None)