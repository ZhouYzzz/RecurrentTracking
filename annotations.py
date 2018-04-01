from xml.etree.ElementTree import parse, Element
from collections import namedtuple
from typing import List


class AnnoMeta(namedtuple('AnnoMeta', ['folder', 'filename', 'size'])):
  pass


class AnnoObj(namedtuple('AnnoObjc', ['trackid', 'name', 'bndbox', 'occluded', 'generated'])):
  pass


class AnnoStream(namedtuple('AnnoStream', ['meta', 'frames', 'bndboxes'])):
  @staticmethod
  def objcs2stream(meta: AnnoMeta, seq_of_objc: List[AnnoObj]):
    raise NotImplementedError


def parse_obj(elem: Element):
  """Parse from raw xml Element (tagged object) to `AnnoObjc`"""
  return AnnoObj(trackid=bytes(elem[0].text, 'utf-8'),
                 name=bytes(elem[1].text, 'utf-8'),
                 bndbox=[int(e.text) for e in elem[2][0:4]],
                 occluded=int(elem[3].text),
                 generated=int(elem[4].text))


def parse_meta(root: Element):
  """Parse from root, return the meta information"""
  return AnnoMeta(folder=bytes(root[0].text, 'utf-8'),
                  filename=bytes(root[1].text, 'utf-8'),
                  size=[int(root[3][0].text), int(root[3][1].text)])


def parse_objs(root: Element):
  """Parse from root, return list of objects"""
  return list(map(parse_obj, root[4::]))  # type: List[AnnoObj]


def parse_anno(root: Element):
  """Parse from root, return all parsed elements

  Returns:
    A Tuple of (AnnoMeta, List[AnnoObj])
  """
  return parse_meta(root), parse_objs(root)


class StreamSeperator(object):
  def __init__(self, dtype):
    self._dtype = dtype
    self._active_stream = dict()
    self._inactive_stream = list()

  def update(self, identified_dict):
    """identified_dict: dict of object {identity: dtype}

    Example: for a dict representing an object `obj`, which is identified with obj['id'],
             StreamSeperator.update({obj['id']: obj for obj in object_list})
    """
    present_ids = identified_dict.keys()
    active_ids = self._active_stream.keys()
    for i in list(present_ids - active_ids):
      self._active_stream[i] = list()
    for i in list(present_ids):
      self._active_stream[i].append(identified_dict[i])
    for i in list(active_ids - present_ids):
      self._inactive_stream.append(self._active_stream.pop(i))

  def close(self):
    active_ids = self._active_stream.keys()
    for i in list(active_ids):
      self._inactive_stream.append(self._active_stream.pop(i))
    return self._inactive_stream


def main():
  pass


if __name__ == '__main__':
    main()