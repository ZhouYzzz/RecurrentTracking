import os, glob, csv
from annotations import parse_annotation_folder

ROOT_DIR = os.path.join(os.path.expanduser('~'), 'ILSVRC2015')


class PHASE:
  TRAIN = 'train'
  EVAL = 'val'
  TEST = 'test'


class ILSVRC2015(object):
  def __init__(self, root_dir):
    if not os.path.exists(root_dir):
      raise NotADirectoryError(root_dir)
    self._root_dir = root_dir

  @property
  def root_dir(self):
    return self._root_dir

  @property
  def imagesets_dir(self):
    return os.path.join(self.root_dir, 'ImageSets', 'VID')

  @property
  def annotations_dir(self):
    return os.path.join(self.root_dir, 'Annotations', 'VID')

  @property
  def data_dir(self):
    return os.path.join(self.root_dir, 'Data', 'VID')

  @property
  def devkit_dir(self):
    return os.path.join(self.root_dir, 'devkit')

  def GetSnippetIDs(self, phase=PHASE.TRAIN):
    files = sorted(glob.glob(os.path.join(self.imagesets_dir, '{}*.txt'.format(phase))))
    if len(files) == 0:
      raise FileExistsError('No ImageSets file exists')
    ids = list()
    for f in files:
      with open(f, 'r') as csvfile:
        ids += [snippet_id for snippet_id, _ in csv.reader(csvfile, delimiter=' ')]
    return ids


def main():
  dataset = ILSVRC2015(root_dir=ROOT_DIR)
  ids = dataset.GetSnippetIDs(PHASE.TRAIN)
  i = 0
  from tqdm import tqdm
  for id in tqdm(ids):
    parse_annotation_folder(os.path.join(dataset.annotations_dir, PHASE.TRAIN, id))



if __name__ == '__main__':
  main()
