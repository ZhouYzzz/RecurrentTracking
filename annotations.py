from xml.etree.ElementTree import parse
from collections import namedtuple
from features import feature_int64


class AnnoMeta(namedtuple('AnnoMeta', ['folder', 'filename', 'size', 'object'])):
  pass

class AnnoObjc(namedtuple('AnnoObjc', ['trackid', 'name', 'bndbox', 'occluded', 'generated'])):
  pass
  # def as_identified_dict(self):
  #   return {self.trackid: self}


def parse_objc(elem):
  return AnnoObjc(trackid=bytes(elem[0].text, 'utf-8'),
                  name=bytes(elem[1].text, 'utf-8'),
                  bndbox=[int(e.text) for e in elem[2][0:4]],
                  occluded=int(elem[3].text),
                  generated=int(elem[4].text))


def parse_meta(elem):
  return AnnoMeta(folder=bytes(elem[0].text, 'utf-8'),
                  filename=bytes(elem[1].text, 'utf-8'),
                  size=[int(elem[3][0].text), int(elem[3][1].text)],
                  object=list(map(parse_objc, elem[4::])))


class StreamSeperator(object):
  def __init__(self, dtype):
    self._dtype = dtype
    self._active_stream = dict()
    self._inactive_stream = list()

  def update(self, identified_dict):
    """d: dict of object {identity: dtype}"""
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
    # tree = parse('../annos/000000.xml')
    tree = parse('/Users/zhouyz/Development/ILSVRC2015/Annotations/VID/train/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00008008/000000.xml')
    root = tree.getroot()
    root[4]
    # print(root[4])
    meta = parse_meta(root)
    ss = StreamSeperator(AnnoObjc)

    for i in range(50):
      tree = parse('/Users/zhouyz/Development/ILSVRC2015/Annotations/VID/train/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00008008/{:06d}.xml'.format(i))
      root = tree.getroot()
      meta = parse_meta(root)
      ss.update({o.trackid: o for o in meta.object})
    # ss.update({o.trackid: o for o in meta.object})
    
    streams = ss.close()
    for s in streams:
      print(len(s))
    # print(meta.object[0].as_identified_dict())
    return
    print(meta)
    f = feature_int64(meta.size)
    print(f)
    for i in meta:
      print(i)
    # print(parse_objc(root[4]))
    # import os


if __name__ == '__main__':
    main()