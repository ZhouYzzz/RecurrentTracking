from xml.etree.ElementTree import parse, Element
from collections import namedtuple
from typing import List
import os
from utils.stream import StreamSeperator


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
  print(parse_annotation_folder(FOLDER))


if __name__ == '__main__':
  main()
