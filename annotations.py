from xml.etree.ElementTree import parse, Element
from collections import namedtuple
from typing import List
import os


class AnnoMeta(namedtuple('AnnoMeta', ['folder', 'filename', 'size'])):
  pass


class AnnoObj(namedtuple('AnnoObj', ['trackid', 'frame', 'name', 'bndbox', 'occluded', 'generated'])):
  pass


class AnnoStream(namedtuple('AnnoStream', ['meta', 'frames', 'bndboxes'])):
  @staticmethod
  def objcs2stream(meta: AnnoMeta, seq_of_objc: List[AnnoObj]):
    raise NotImplementedError

  def __new__(cls, meta: AnnoMeta, objs: List[AnnoObj]):
    frames = [o.frame for o in objs]
    bndboxes = [o.bndbox for o in objs]
    return super(AnnoStream, cls).__new__(cls, meta=meta, frames=frames, bndboxes=bndboxes)


def fixed_length_slice_with_pad(x: List, s: int, l: int):
  lx = len(x)  # length of list `x`
  if s > lx:
    raise ValueError('s({}) exceeds array length ({})'.format(s, lx))
  e = s + l    # end
  if e < 0:
    raise ValueError('e({}) = s({}) + l({}) should be non-negative'.format(e, s, l))
  (sp, s) = (-s, 0) if s < 0 else (0, s)
  (ep, e) = (e - lx, lx) if e > lx else (0, e)
  return [x[0] for _ in range(sp)] + x[slice(s, e)] + [x[-1] for _ in range(ep)]

def parse_obj(elem: Element, frame: int):
  """Parse from raw xml Element (tagged object) to `AnnoObjc`"""
  return AnnoObj(trackid=bytes(elem[0].text, 'utf-8'),
                 frame=int(frame),
                 name=bytes(elem[1].text, 'utf-8'),
                 bndbox=[int(e.text) for e in elem[2][0:4]],
                 occluded=int(elem[3].text),
                 generated=int(elem[4].text))


def parse_meta(root: Element):
  """Parse from root, return the meta information"""
  return AnnoMeta(folder=bytes(root[0].text, 'utf-8'),
                  filename=bytes(root[1].text, 'utf-8'),
                  size=[int(root[3][0].text), int(root[3][1].text)])


def parse_objs(root: Element, frame: int):
  """Parse from root, return list of objects"""
  return list(map(lambda e: parse_obj(e, frame=frame), root[4::]))  # type: List[AnnoObj]


def parse_anno(root: Element, frame: int):
  """Parse from root, return all parsed elements

  Returns:
    A Tuple of (AnnoMeta, List[AnnoObj])
  """
  return parse_meta(root), parse_objs(root, frame=frame)


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


def construct_stream(meta: AnnoMeta, objstream: List[AnnoObj]):
  return AnnoStream(meta=meta, frames=[o.frame for o in objstream], bndboxes=[o.bndbox for o in objstream])


def split_stream(stream: AnnoStream):
  pass


def main():
  x = list(range(10))
  print(x)
  try:
    print(fixed_length_slice_with_pad(x, -20, 6))
  except:
    print('Left Error')
  print(fixed_length_slice_with_pad(x, -2, 6))
  print(fixed_length_slice_with_pad(x, 0, 6))
  print(fixed_length_slice_with_pad(x, 2, 6))
  print(fixed_length_slice_with_pad(x, -1, 12))
  print(fixed_length_slice_with_pad(x, 6, 6))
  try:
    print(fixed_length_slice_with_pad(x, 12, 6))
  except:
    print('Right Error')


def parse_annotation_folder(folder: str):
  num_xml_files = len([f for f in os.listdir(folder) if f.endswith('.xml')])
  stream_seperator = StreamSeperator(AnnoObj)
  for i in range(num_xml_files):
    root = parse(os.path.join(folder, '{:06d}.xml'.format(i))).getroot()
    if not i:
      meta = parse_meta(root)
    objs = parse_objs(root, frame=i)
    stream_seperator.update({o.trackid: o for o in objs})
  streams = stream_seperator.close()
  return [construct_stream(meta=meta, objstream=stream) for stream in streams]


def main():
  FOLDER = '/home/zhouyz/ILSVRC2015/Annotations/VID/val/ILSVRC2015_val_00030000'
  print(parse_annotation_folder(FOLDER))


if __name__ == '__main__':
    main()