from easydict import EasyDict as edict
import xml.etree.ElementTree as ET

def parsexml(filename):
    tree = ET.parse(filename)
    size = tree.find('size')
    objs = tree.findall('object')
    size = [int(c.text) for c in size.getchildren()]
    objs = [ele2dict(c) for c in objs]
    d = edict()
    d.size = size
    d.objs = objs
    return d

def ele2dict(c):
    d = edict()
    d.trackid = int(c.find('trackid').text)
    d.bndbox = [int(i.text) for i in c.find('bndbox').getchildren()]
    d.occluded = bool(c.find('occluded').text)
    d.generated = bool(c.find('generated').text)
    return d

