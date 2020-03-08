import argparse, json, os, glob
from collections import Counter, defaultdict
import PIL

import json, os, random, math
from collections import Counter, defaultdict

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import numpy as np
import PIL
from skimage.transform import resize as imresize
import pycocotools.mask as mask_utils

from sg2im.data.utils import imagenet_preprocess, Resize

import copy


COCO_DIR = 'datasets/coco/annotations/'

parser = argparse.ArgumentParser()
parser.add_argument('--instance_json', \
    default=os.path.join(COCO_DIR, 'instances_train2017.json'))
parser.add_argument('--sg_json', \
    default=os.path.join(COCO_DIR, 'sg_train2017.json'))

parser.add_argument('--error_img_ids', \
    default="error_imgids.txt")

def combine_json():
    result = []
    for i in range(26):
        f = COCO_DIR + "sg_new/" + "sg_val2017_" + str(i+1) + "_new.json"
        print(f)
        json_data = []
        with open(f, "rb") as infile:
            json_data = json.load(infile)
            print(len(json_data))
            result.extend(json_data)
    print(result[0])
    with open(COCO_DIR + "sg_val2017.json", "w") as outfile:
        json.dump(result, outfile)

def read_json(json_file):
    with open(json_file, 'r') as f:
      json_data = json.load(f)
    # for key, value in json_data.items() :
    #     print (key)

    read_instance(json_data)

def read_coco(instance_json, sg_json):
    with open(instance_json, 'r') as f:
      instances_data = json.load(f)

    image_ids = []
    image_id_to_filename = {}
    image_id_to_size = {}

    for image_data in instances_data['images']:
      image_id = image_data['id']
      filename = image_data['file_name']
      width = image_data['width']
      height = image_data['height']
      image_ids.append(image_id)
      image_id_to_filename[image_id] = filename
      image_id_to_size[image_id] = (width, height)

    vocab = {
      'object_name_to_idx': {},
      'object_idx_to_name': [],
      'pred_name_to_idx': {},
      'pred_idx_to_name': []
    }

    # Add object data from instances
    image_id_to_objects = defaultdict(list)
    image_id_to_relationships = defaultdict(list)

    sg_data = []
    with open(sg_json, 'r') as f:
      sg_data = json.load(f)
    
    object_name_counter = Counter()
    pred_counter = Counter()
    for sg_obj in sg_data:
      image_id = sg_obj['image_id']
      if image_id not in image_ids:
        continue
      # add the objects to vocab
      names = set()
      for obj in sg_obj['objects']:
        names.add(obj)
      object_name_counter.update(names)
      # add the predicates
      preds = set()
      for rel in sg_obj['relationships']:
        preds.add(rel[1])
      pred_counter.update(preds)
      # add the objects to the image_id_to_objects
      image_id_to_objects[image_id].append(sg_obj['objects'])
      image_id_to_relationships[image_id].append(sg_obj['relationships'])
    
    
    # print(object_name_counter)
    object_names = ['__image__']
    for name, count in object_name_counter.most_common():
      object_names.append(name)
    print('Found %d object categories.' % (len(object_names)))
    
    object_name_to_idx = {}
    object_idx_to_name = []
    for idx, name in enumerate(object_names):
      object_name_to_idx[name] = idx
      object_idx_to_name.append(name)

    # print(pred_counter)
    pred_names = ['__in_image__']
    for pred, count in pred_counter.most_common():
      pred_names.append(pred)
    print('Found %d relationship types' % (len(pred_names)))

    pred_name_to_idx = {}
    pred_idx_to_name = []
    for idx, name in enumerate(pred_names):
      pred_name_to_idx[name] = idx
      pred_idx_to_name.append(name)

    vocab['object_name_to_idx'] = object_name_to_idx
    vocab['object_idx_to_name'] = object_idx_to_name
    vocab['pred_name_to_idx'] = pred_name_to_idx
    vocab['pred_idx_to_name'] = pred_idx_to_name

def read_images(instances_json, sg_json):
    # for 5 different captions
    suffix = ['a', 'b', 'c', 'd', 'e']
    # error = [77137, 292505, 408099, 80819, 381972, 493610, 133225, 470112]
    error = [178971, 375882, 231677]
    error_image_ids = []
    for id in error:
      for suf in suffix:
        error_image_ids.append(str(id) + suf)

    min_object_size=0.02
    with open(instances_json, 'r') as f:
      instances_data = json.load(f)

    image_ids = []
    original_image_ids = []
    image_id_to_filename = {}
    image_id_to_size = {}
    for image_data in instances_data['images']:
      image_id = image_data['id']
      filename = image_data['file_name']
      width = image_data['width']
      height = image_data['height']
      original_image_ids.append(image_id)
      image_id_to_size[image_id] = (width, height)
      for suf in suffix:
        new_image_id = str(image_id) + suf
        image_ids.append(new_image_id)
        image_id_to_filename[new_image_id] = filename
        

    print("Total inital images %d" % (len(image_ids)))

    vocab = {
      'object_name_to_idx': {}    
    }

    object_idx_to_name = {}
    all_instance_categories = []
    for category_data in instances_data['categories']:
      category_id = category_data['id']
      category_name = category_data['name']
      all_instance_categories.append(category_name)
      object_idx_to_name[category_id] = category_name
      vocab['object_name_to_idx'][category_name] = category_id

    # COCO category labels start at 1, so use 0 for __image__
    vocab['object_name_to_idx']['__image__'] = 0

    # Build object_idx_to_name
    name_to_idx = vocab['object_name_to_idx']
    assert len(name_to_idx) == len(set(name_to_idx.values()))
    max_object_idx = max(name_to_idx.values())
    idx_to_name = ['NONE'] * (1 + max_object_idx)
    for name, idx in vocab['object_name_to_idx'].items():
      idx_to_name[idx] = name
    vocab['object_idx_to_name'] = idx_to_name

    # Add object data from instances
    image_id_to_objects = defaultdict(list)
    image_id_to_objects_names = defaultdict(list)
    image_id_to_objects_bbox = defaultdict(dict)
    image_id_to_objects_seg = defaultdict(dict)
    for object_data in instances_data['annotations']:
      image_id = object_data['image_id']
      _, _, w, h = object_data['bbox']
      W, H = image_id_to_size[image_id]
      box_area = (w * h) / (W * H)
      box_ok = box_area > min_object_size
      object_name = object_idx_to_name[object_data['category_id']]
      if box_ok:
        for suf in suffix:
          new_image_id = str(image_id) + suf
          image_id_to_objects_names[new_image_id].append(object_name)
          image_id_to_objects[new_image_id].append(object_data)
          image_id_to_objects_bbox[new_image_id][object_name] = object_data['bbox']
          image_id_to_objects_seg[new_image_id][object_name] = object_data['segmentation']


    # Add object data from instances
    image_id_to_sg_objects = defaultdict(list)
    image_id_to_relationships = defaultdict(list)

    sg_data = []
    with open(sg_json, 'r') as f:
      sg_data = json.load(f)
    
    object_name_counter = Counter()
    pred_counter = Counter()
    caption_id_map = defaultdict(list)

    for sg_obj in sg_data:
      image_id = sg_obj['image_id']
      if image_id not in error:
        continue
      image_caption_id = find_available_caption_id(image_id, suffix, caption_id_map)
      # add the objects to vocab
      sg_obj_list = []
      names = []
      match, idx_map = match_objs(sg_obj['objects'], \
        image_id_to_objects_names[image_caption_id])
      for key, value in match.items():
        # add the matched coco object to names
        print(key, value)
        names.append(value)
        # add the matched coco object to image id
        image_id_to_sg_objects[image_caption_id].append(value)
        # add the matched sg object to sg_obj_list
        sg_obj_list.append(key)
      object_name_counter.update(names)
      for temp_name in image_id_to_objects_names[image_caption_id]:
        if temp_name not in image_id_to_sg_objects[image_caption_id]:
          image_id_to_sg_objects[image_caption_id].append(temp_name)
      print(image_caption_id)
      print("==sg_objs")
      print(sg_obj['objects'])
      print("==image_id_to_objects_names[image_id]")
      print(image_id_to_objects_names[image_caption_id])
      print("==sg_obj_list")
      print(sg_obj_list)
      print("==image_id_to_sg_objects[image_id]")
      print(image_id_to_sg_objects[image_caption_id])
      
      # add the predicates
      preds = set()
      for rel in sg_obj['relationships']:
        newRel = []
        print(rel)
        if (sg_obj['objects'][rel[0]] in sg_obj_list) \
          and (sg_obj['objects'][rel[2]] in sg_obj_list):
          preds.add(rel[1])
          s_idx = rel[0]
          newRel.append(idx_map[s_idx])
          newRel.append(rel[1])
          o_idx = rel[2]
          newRel.append(idx_map[o_idx])      
          image_id_to_relationships[image_caption_id].append(newRel)
      print("==image_id_to_relationships")
      print(image_id_to_relationships[image_caption_id])
      pred_counter.update(preds)

    object_names = ['__image__']
    for name, count in object_name_counter.most_common():
      object_names.append(name)
    print('Found %d object categories.' % (len(object_names)))

    min_objects_per_image=3
    max_objects_per_image=8
    # Prune images that have too few or too many objects
    new_image_ids = []
    total_objs = 0
    for image_id in image_ids:
      num_objs = len(image_id_to_sg_objects[image_caption_id])
      total_objs += num_objs
      if min_objects_per_image <= num_objs <= max_objects_per_image:
        new_image_ids.append(image_id)
    image_ids = new_image_ids
    print('After pruning, %d images left.' % (len(image_ids)))
    
    pred_names =  [
    '__in_image__',
    'left of',
    'right of',
    'above',
    'below',
    'inside',
    'surrounding',
    ]
    for pred, count in pred_counter.most_common():
      pred_names.append(pred)
    print('Found %d relationship types' % (len(pred_names)))

    pred_name_to_idx = {}
    pred_idx_to_name = []
    for idx, name in enumerate(pred_names):
      pred_name_to_idx[name] = idx
      pred_idx_to_name.append(name)

    vocab['pred_name_to_idx'] = pred_name_to_idx
    vocab['pred_idx_to_name'] = pred_idx_to_name

    ##################################################################
    ########### get item #############################################
    image_size=(64, 64)
    mask_size=16
    image_dir = "datasets/coco/images/train2017"
    image_id = error_image_ids[0]
    print("===========get item========" + str(image_id))
    transform = [Resize(image_size), T.ToTensor()]
    transform = T.Compose(transform)
    filename = image_id_to_filename[image_id]
    image_path = os.path.join(image_dir, filename)
    with open(image_path, 'rb') as f:
      with PIL.Image.open(f) as image:
        WW, HH = image.size
        image = transform(image.convert('RGB'))

    
    H, W = image_size
    objs, boxes, masks = [], [], []
    for object_name in image_id_to_sg_objects[image_id]:
      objs.append(vocab['object_name_to_idx'][object_name])
      # bbox
      x, y, w, h = image_id_to_objects_bbox[image_id][object_name]
      x0 = x / WW
      y0 = y / HH
      x1 = (x + w) / WW
      y1 = (y + h) / HH
      boxes.append(torch.FloatTensor([x0, y0, x1, y1]))
      
      # This will give a numpy array of shape (HH, WW)
      mask = seg_to_mask(image_id_to_objects_seg[image_id][object_name], WW, HH)

      # Crop the mask according to the bounding box, being careful to
      # ensure that we don't crop a zero-area region
      mx0, mx1 = int(round(x)), int(round(x + w))
      my0, my1 = int(round(y)), int(round(y + h))
      mx1 = max(mx0 + 1, mx1)
      my1 = max(my0 + 1, my1)
      mask = mask[my0:my1, mx0:mx1]
      mask = imresize(255.0 * mask, (mask_size, mask_size),
                      mode='constant')
      mask = torch.from_numpy((mask > 128).astype(np.int64))
      masks.append(mask)
    
    print(objs)
    print(len(objs))
    print (len(masks))
    assert len(masks) == len(objs)
    assert len(boxes) == len(objs)
    # Add dummy __image__ object
    objs.append(vocab['object_name_to_idx']['__image__'])
    boxes.append(torch.FloatTensor([0, 0, 1, 1]))
    masks.append(torch.ones(mask_size, mask_size).long())

    objs = torch.LongTensor(objs)
    boxes = torch.stack(boxes, dim=0)
    masks = torch.stack(masks, dim=0)

    box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # Compute centers of all objects
    obj_centers = []
    _, MH, MW = masks.size()
    for i, obj_idx in enumerate(objs):
      x0, y0, x1, y1 = boxes[i]
      mask = (masks[i] == 1)
      xs = torch.linspace(x0, x1, MW).view(1, MW).expand(MH, MW)
      ys = torch.linspace(y0, y1, MH).view(MH, 1).expand(MH, MW)
      if mask.sum() == 0:
        mean_x = 0.5 * (x0 + x1)
        mean_y = 0.5 * (y0 + y1)
      else:
        mean_x = xs[mask].mean()
        mean_y = ys[mask].mean()
      obj_centers.append([mean_x, mean_y])
    obj_centers = torch.FloatTensor(obj_centers)

    # Add triples
    triples = []

    num_objs = objs.size(0)
    __image__ = vocab['object_name_to_idx']['__image__']
    real_objs = []
    if num_objs > 1:
      real_objs = (objs != __image__).nonzero().squeeze(1)
    for cur in real_objs:
      choices = [obj for obj in real_objs if obj != cur]
      if len(choices) == 0:
        break
      other = random.choice(choices)
      if random.random() > 0.5:
        s, o = cur, other
      else:
        s, o = other, cur

      # Check for inside / surrounding
      sx0, sy0, sx1, sy1 = boxes[s]
      ox0, oy0, ox1, oy1 = boxes[o]
      d = obj_centers[s] - obj_centers[o]
      theta = math.atan2(d[1], d[0])

      if sx0 < ox0 and sx1 > ox1 and sy0 < oy0 and sy1 > oy1:
        p = 'surrounding'
      elif sx0 > ox0 and sx1 < ox1 and sy0 > oy0 and sy1 < oy1:
        p = 'inside'
      elif theta >= 3 * math.pi / 4 or theta <= -3 * math.pi / 4:
        p = 'left of'
      elif -3 * math.pi / 4 <= theta < -math.pi / 4:
        p = 'above'
      elif -math.pi / 4 <= theta < math.pi / 4:
        p = 'right of'
      elif math.pi / 4 <= theta < 3 * math.pi / 4:
        p = 'below'
      p = vocab['pred_name_to_idx'][p]
      triples.append([s, p, o])

    for rel in image_id_to_relationships[image_id]:
      print(rel)
      s = int(rel[0])
      p = vocab['pred_name_to_idx'].get(rel[1], None)
      o = int(rel[2])
      if s is not None and o is not None and p is not None:
        triples.append([s, p, o])
    
    # Add __in_image__ triples
    O = objs.size(0)
    in_image = vocab['pred_name_to_idx']['__in_image__']
    for i in range(O - 1):
      triples.append([i, in_image, O - 1])
    
    print(triples)
    triples = torch.LongTensor(triples)

def seg_to_mask(seg, width=1.0, height=1.0):
  """
  Tiny utility for decoding segmentation masks using the pycocotools API.
  """
  if type(seg) == list:
    rles = mask_utils.frPyObjects(seg, height, width)
    rle = mask_utils.merge(rles)
  elif type(seg['counts']) == list:
    rle = mask_utils.frPyObjects(seg, height, width)
  else:
    rle = seg
  return mask_utils.decode(rle)


def find_available_caption_id(image_id, suffix, caption_id_map):
  # find the available image_caption_id
  for suf in suffix:
    new_caption_id = str(image_id) + suf
    if new_caption_id not in caption_id_map[image_id]:
      caption_id_map[image_id].append(new_caption_id)
      return new_caption_id


def read_instance(instances_data):
    print(len(instances_data['images']))
    for image_data in instances_data['images']:
        print("============= images ==============")
        for key, value in image_data.items():
            print(key, value)
        break

    print(len(instances_data['annotations']))
    for object_data in instances_data['annotations']:
        print("============= annotations ==============")
        for key, value in object_data.items():
            print(key, value)
        break

    for category_data in instances_data['categories']:
        print("============= categories ==============")
        for key, value in category_data.items():
            print(key, value)
        break

def match_objs(sg_objs, coco_objs):
    match = {}
    idx_map = {}
    sg_idx = 0
    new_sg_idx = 0
    for sg_obj in sg_objs:
      for coco_obj in coco_objs:
        if ((sg_obj in coco_obj) or (coco_obj in sg_obj)):
          match[sg_obj] = coco_obj
          break
      if match.get(sg_obj, None) != None:
        idx_map[sg_idx] = new_sg_idx
        new_sg_idx += 1
      sg_idx += 1
      
    return match, idx_map 

def read_error_img_ids(error_imgs_file):
  with open(error_imgs_file, 'r') as f:
      img_ids = f.readlines() 
  
  new_img_ids = []
  for id in img_ids:
    len_id = len(id)
    if (len_id > 2):
      new_img_ids.append(id[0:len_id//2])
  
  return new_img_ids


if __name__ == '__main__':
  args = parser.parse_args()
#   read_coco(args.instance_json, args.sg_json)
#   read_json(args.instance_json)
#   combine_json()
  # read_images(args.instance_json, args.sg_json)
  read_error_img_ids(args.error_img_ids)
  # a = ['aaa', 'aaa', 'b', 'c', 'd']
  # b = ['aaa', 'c', 'd']
  # match, idx_map = match(a, b)
  # print(match)
  # print(idx_map)