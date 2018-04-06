from xml.etree.ElementTree import parse, Element
from collections import namedtuple
from typing import List
import os
from utils.stream import StreamSeperator
from utils.slice import fixed_length_slice_with_pad
import tensorflow as tf
from features import *


class AnnoMeta(namedtuple('AnnoMeta', ['folder', 'filename', 'size'])):
  pass


class AnnoObj(namedtuple('AnnoObj', ['trackid', 'frame', 'name', 'bndbox', 'occluded', 'generated'])):
  pass


class AnnoStream(namedtuple('AnnoStream', ['meta', 'length', 'frames', 'bndboxes'])):
  def __new__(cls, meta: AnnoMeta, objs: List[AnnoObj]):
    frames = [o.frame for o in objs]
    bndboxes = [o.bndbox for o in objs]
    return super(AnnoStream, cls).__new__(cls, meta=meta, length=len(objs), frames=frames, bndboxes=bndboxes)

  def substream(self, s: int, l: int):
    """Extract a `l`-lengthed substream which starts at `s`, duplicated padding is performed"""
    frames = fixed_length_slice_with_pad(self.frames, s, l)
    bndboxes = fixed_length_slice_with_pad(self.bndboxes, s, l)
    return super().__new__(type(self), meta=self.meta, length=l, frames=frames, bndboxes=bndboxes)

  def splitIntoStreams(self, n: int, l: int):
    """Split the whole AnnoStream into `n` same-lengthed substreams"""
    _interval = max(self.length, 0) // n
    ss = map(lambda x: _interval * x, range(0, n, 1))
    return [self.substream(s=s, l=l) for s in ss]

  def serializeToTFSequenceExample(self):
    """Serialize AnnoStream object to TF.SequenceExample instance"""
    return tf.train.SequenceExample(
      context=tf.train.Features(feature={
        'folder': feature_bytes(self.meta.folder),
        'size': feature_int64(self.meta.size),
        'length': feature_int64(self.length)
      }),
      feature_lists=tf.train.FeatureLists(feature_list={
        'frames': feature_list_int64(self.frames),
        'bndboxes': feature_list_int64(self.bndboxes)
      })
    )


def parse_obj(elem: Element, frame: int):
  """Parse from raw xml Element (tagged object) to `AnnoObjc`"""
  return AnnoObj(trackid=bytes(elem[0].text, 'utf-8'),
                 frame=[int(frame)],  # 1-D scalar feature, should be represented as a list
                 name=bytes(elem[1].text, 'utf-8'),
                 bndbox=[int(e.text) for e in elem[2][0:4]],  # xmax, xmin, ymax, ymin
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


def parse_annotation_folder(folder: str):
  """Parse a full annotation folder to seperated video streams, each is a continuous, single-objected sequence"""
  num_xml_files = len([f for f in os.listdir(folder) if f.endswith('.xml')])
  stream_seperator = StreamSeperator(AnnoObj)
  for i in range(num_xml_files):
    root = parse(os.path.join(folder, '{:06d}.xml'.format(i))).getroot()
    if not i:
      meta = parse_meta(root)
    objs = parse_objs(root, frame=i)
    stream_seperator.update({o.trackid: o for o in objs})
  streams = stream_seperator.close()
  return [AnnoStream(meta=meta, objs=stream) for stream in streams]


def main():
  FOLDER = '/home/zhouyz/ILSVRC2015/Annotations/VID/val/ILSVRC2015_val_00030000'
  streams = parse_annotation_folder(FOLDER)
  stream = streams[0]
  print(stream.length, stream.substream(0, 10))
  print(stream.substream(0, 4).serializeToTFSequenceExample())#.SerializeToString())
  print(stream.splitIntoStreams(2, 2))


if __name__ == '__main__':
  main()
