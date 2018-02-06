from xml.etree.ElementTree import parse
import os
import glob
import random


class Snippet(object):
  def __init__(self):
    self._frames = list()
    self._bndboxes = list()

  def append(self, frame, bndbox):
    self._frames.append(frame)
    self._bndboxes.append(bndbox)

  @property
  def frames(self):
    return self._frames

  @property
  def bndboxes(self):
    return self._bndboxes

  @property
  def length(self):
    return len(self._frames)

  @property
  def current_frame(self):
    return self._frames[-1]

  def subsnippet(self, length=32):
    _length = min(self.length, length)
    start = random.randint(0, self.length - _length)
    s_pad = []
    if length > _length:
      s_pad = self.padsnippet(length=length-_length, index=start)
    stop = start + _length
    s = Snippet()
    s._frames = self._frames[slice(start, stop)]
    s._bndboxes = self._bndboxes[slice(start, stop)]
    s_pad += s
    return s_pad

  def padsnippet(self, length=32, index=0):
    s = Snippet()
    s._frames = self._frames[[index for _ in range(length)]]
    s._bndboxes = self._bndboxes[[index for _ in range(length)]]
    return s
  
  def appendsnippet(self, s):
    assert isinstance(s, Snippet)
    self._frames.append(s._frames)
    self._bndboxes.append(s._bndboxes)

  def todict(self):
    return {'length': self.length,
            'frames': self.frames,
            'bndboxes': self.bndboxes}


class SnippetRegister(object):
  def __init__(self):
    self._ongoing_snippets = dict()
    self._ended_snippets = list()

  def register(self, frame, shots):
    present_trackids = [shot['trackid'] for shot in shots]
    for shot in shots:
      trackid = shot['trackid']
      bndbox = shot['bndbox']
      if trackid not in self._ongoing_snippets.keys():
        # register new object when it first appears
        self._ongoing_snippets[trackid] = Snippet()

      # track existing object
      self._ongoing_snippets[trackid].append(frame, bndbox)

    for trackid in self._ongoing_snippets:
      if trackid not in present_trackids:
        # close ended object sequence
        self._ended_snippets.append(self._ongoing_snippets.pop(trackid))

  def close(self):
    for trackid in self._ongoing_snippets.keys():
      self._ended_snippets.append(self._ongoing_snippets.pop(trackid))
    return self._ended_snippets


def parse_snippet_annotations(snippet_anno_dir):
  anno_filenames = glob.glob(os.path.join(snippet_anno_dir, '*.xml'))
  snippet_length = len(anno_filenames)
  meta = parse_annotation_file(os.path.join(snippet_anno_dir, '000000.xml'))
  register = SnippetRegister()
  register.register(0, meta['object'])
  for i in range(1, snippet_length):
    anno = parse_annotation_file(os.path.join(snippet_anno_dir, '{:06d}.xml'.format(i)))
    register.register(i, anno['object'])
  snippets = register.close()
  return {'snippet_id': meta['folder'],
          'subsnippets': map(lambda s: s.todict(), snippets)}


def parse_annotation_file(filename):
  return parse_annotation(parse(filename))


def parse_annotation(tree):
  """
  Parse the annotation tree for ILSVRC2015 VID challenge
  @param tree An ElementTree instance
  @return A parsed dict
  """
  def parse_object(element):
    _trackid = element[0]
    _name = element[1]
    _bndbox = element[2]
    _occluded = element[3]
    _generated = element[4]
    return {'trackid': _trackid.text,
            'name': _name.text,
            'bndbox': [int(e.text) for e in _bndbox[0:4]], # xmax, xmin, ymax, ymin
            'occluded': int(_occluded.text),
            'generated': int(_generated.text)}
  _root = tree.getroot()
  _folder = _root[0]
  _filename = _root[1]
  _source = _root[2]
  _size = _root[3]
  _object = _root[4::]
  return {'folder': _folder.text,
          'filename': _filename.text,
          'size': {'width': int(_size[0].text), 'height': int(_size[1].text)},
          'object': map(parse_object, _object)}


def main():
  annos = parse_snippet_annotations(snippet_anno_dir='Annotations/VID/train/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00008008')
  print(annos)

if __name__ == '__main__':
  main()