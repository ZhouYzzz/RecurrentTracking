from typing import Any, Dict, List

class Snippet(object):
  """
  `Snippet` is a container to store sequential data.
  Several `T`-sized lists are used to record different properties respectively,
  e.g. temperature, weather, wind speed. e.t.c.
  """
  def __init__(self, **kwargs) -> None:
    self._entities = dict() # type: Dict[str, Any]
    self._inited = False
    self._length = 0
    if len(kwargs) == 0:
      return
    self._init_with_keys(*kwargs.keys())
    self.append(**kwargs)

  def _init_with_keys(self, *keys) -> None:
    for k in keys:
      self._entities[k] = list()
    self._inited = True

  def append(self, **kwargs) -> None:
    """
    `append(foo=0, bar='sunny', ..., **kwargs)`
    """
    if not self._inited:
      self._init_with_keys(*kwargs.keys())
    for k in self._entities.keys():
      self._entities[k].append(kwargs[k])
    self._length += 1

  @property
  def entities(self):
    return self._entities

  @property
  def length(self):
    return self._length

  def sub(self, start: int, length: int, padding: bool = True):
    """
    `sub` selects the length `length` subset of the whole snippet begins at `start`.
    
    `start: int` the start indice
    `length: int` the length of the subsnippet
    `padding: bool = True` whether to pad the subsnipet if exceeds the range
    """
    if not padding:
      raise NotImplementedError
    _pad_s = max(0 - start, 0)
    _pad_e = max(start + length - self._length, 0)
    _length = length - _pad_s - _pad_e
    _start = min(max(start, 0), self._length)
    subinds = [0] * _pad_s + list(range(_start, _start + _length)) + [self._length - 1] * _pad_e
    assert(len(subinds) == length)
    subsnippet = Snippet()
    subsnippet._inited = True
    subsnippet._length = length
    for k, v in self._entities.items():
      subsnippet._entities[k] = [v[i] for i in subinds]
    return subsnippet

  def split(self, n: int, l: int, padding: bool = True):
    """
    `split` splits the `Snippet` into `n` `Snippet` of length `l` each.

    `n: int` num of subsnippet to get
    `l: int` the length of each subsnippet
    `padding: bool = True` whether to pad each subsnipet if exceeds the range
    """
    if not padding:
      raise NotImplementedError
    _interval = max(self._length, 0) // n
    _starts = map(lambda x: _interval * x, range(0, n, 1))
    return [self.sub(start=s, length=l, padding=padding) for s in _starts]


def main():
  s = Snippet()
  for i in range(100):
    s.append(frame=i, location=[i,i])
  ss = s.split(2, 10)
  for i in ss:
    print(i.entities)

if __name__ == '__main__':
  main()