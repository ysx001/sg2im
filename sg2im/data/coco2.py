#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json, os, random, math
from collections import Counter, defaultdict

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import numpy as np
import PIL
from skimage.transform import resize as imresize
import pycocotools.mask as mask_utils

from .utils import imagenet_preprocess, Resize


class CocoSGDataset(Dataset):
  def __init__(self, image_dir, instances_json, sg_json, stuff_json=None,
               stuff_only=True, image_size=(64, 64), mask_size=16,
               normalize_images=True, max_samples=None,
               include_relationships=True, min_object_size=0.02,
               min_objects_per_image=3, max_objects_per_image=8,
               include_other=False, instance_whitelist=None, stuff_whitelist=None):
    """
    A PyTorch Dataset for loading Coco and Coco-Stuff annotations and converting
    them to scene graphs on the fly.

    Inputs:
    - image_dir: Path to a directory where images are held
    - instances_json: Path to a JSON file giving COCO annotations
    - stuff_json: (optional) Path to a JSON file giving COCO-Stuff annotations
    - stuff_only: (optional, default True) If True then only iterate over
      images which appear in stuff_json; if False then iterate over all images
      in instances_json.
    - image_size: Size (H, W) at which to load images. Default (64, 64).
    - mask_size: Size M for object segmentation masks; default 16.
    - normalize_image: If True then normalize images by subtracting ImageNet
      mean pixel and dividing by ImageNet std pixel.
    - max_samples: If None use all images. Other wise only use images in the
      range [0, max_samples). Default None.
    - include_relationships: If True then include spatial relationships; if
      False then only include the trivial __in_image__ relationship.
    - min_object_size: Ignore objects whose bounding box takes up less than
      this fraction of the image.
    - min_objects_per_image: Ignore images which have fewer than this many
      object annotations.
    - max_objects_per_image: Ignore images which have more than this many
      object annotations.
    - include_other: If True, include COCO-Stuff annotations which have category
      "other". Default is False, because I found that these were really noisy
      and pretty much impossible for the system to model.
    - instance_whitelist: None means use all instance categories. Otherwise a
      list giving a whitelist of instance category names to use.
    - stuff_whitelist: None means use all stuff categories. Otherwise a list
      giving a whitelist of stuff category names to use.
    """
    super(Dataset, self).__init__()

    if stuff_only and stuff_json is None:
      print('WARNING: Got stuff_only=True but stuff_json=None.')
      print('Falling back to stuff_only=False.')

    self.image_dir = image_dir
    self.mask_size = mask_size
    self.max_samples = max_samples
    self.normalize_images = normalize_images
    self.include_relationships = include_relationships
    self.set_image_size(image_size)

    # for 5 different captions
    self.suffix = ['a', 'b', 'c', 'd', 'e']


    with open(instances_json, 'r') as f:
      instances_data = json.load(f)

    self.image_ids = []
    self.original_image_ids = []
    self.image_id_to_filename = {}
    self.image_id_to_size = {}
    for image_data in instances_data['images']:
      image_id = image_data['id']
      filename = image_data['file_name']
      width = image_data['width']
      height = image_data['height']
      self.original_image_ids.append(image_id)
      self.image_id_to_filename[image_id] = filename
      self.image_id_to_size[image_id] = (width, height)
      for suf in self.suffix:
        new_image_id = str(image_id) + suf
        self.image_ids.append(new_image_id)
        self.image_id_to_filename[new_image_id] = filename

    print("Total inital images %d" % (len(self.image_ids)))

    self.vocab = {
      'object_name_to_idx': {}    
    }

    object_idx_to_name = {}
    all_instance_categories = []
    for category_data in instances_data['categories']:
      category_id = category_data['id']
      category_name = category_data['name']
      all_instance_categories.append(category_name)
      object_idx_to_name[category_id] = category_name
      self.vocab['object_name_to_idx'][category_name] = category_id

    # COCO category labels start at 1, so use 0 for __image__
    self.vocab['object_name_to_idx']['__image__'] = 0

    # Build object_idx_to_name
    name_to_idx = self.vocab['object_name_to_idx']
    assert len(name_to_idx) == len(set(name_to_idx.values()))
    max_object_idx = max(name_to_idx.values())
    idx_to_name = ['NONE'] * (1 + max_object_idx)
    for name, idx in self.vocab['object_name_to_idx'].items():
      idx_to_name[idx] = name
    self.vocab['object_idx_to_name'] = idx_to_name

    # Add object data from instances
    self.image_id_to_objects = defaultdict(list)
    self.image_id_to_objects_names = defaultdict(list)
    self.image_id_to_objects_bbox = defaultdict(dict)
    self.image_id_to_objects_seg = defaultdict(dict)
    for object_data in instances_data['annotations']:
      image_id = object_data['image_id']
      _, _, w, h = object_data['bbox']
      W, H = self.image_id_to_size[image_id]
      box_area = (w * h) / (W * H)
      box_ok = box_area > min_object_size
      object_name = object_idx_to_name[object_data['category_id']]
      if box_ok:
        for suf in self.suffix:
          new_image_id = str(image_id) + suf
          self.image_id_to_objects_names[new_image_id].append(object_name)
          self.image_id_to_objects[new_image_id].append(object_data)
          self.image_id_to_objects_bbox[new_image_id][object_name] = object_data['bbox']
          self.image_id_to_objects_seg[new_image_id][object_name] = object_data['segmentation']

    # Add object data from instances
    self.image_id_to_sg_objects = defaultdict(list)
    self.image_id_to_relationships = defaultdict(list)

    sg_data = []
    with open(sg_json, 'r') as f:
      sg_data = json.load(f)
    
    object_name_counter = Counter()
    pred_counter = Counter()
    self.caption_id_map = defaultdict(list)

    for sg_obj in sg_data:
      image_id = sg_obj['image_id']
      if image_id not in self.original_image_ids:
        continue

      image_caption_id = self.find_available_caption_id(image_id)

      # add the objects to vocab
      sg_obj_list = []
      names = set()
      match, idx_map = self.match_objs(sg_obj['objects'], \
        self.image_id_to_objects_names[image_caption_id])
      for key, value in match.items():
        # add the matched coco object to names
        names.add(value)
        # add the matched coco object to image id
        self.image_id_to_sg_objects[image_caption_id].append(value)
        # add the matched sg object to sg_obj_list
        sg_obj_list.append(key)
      object_name_counter.update(names)
      for obj in self.image_id_to_objects_names[image_caption_id]:
        if obj not in self.image_id_to_sg_objects[image_caption_id]:
          self.image_id_to_sg_objects[image_caption_id].append(obj)
  
      # add the predicates
      preds = set()
      for rel in sg_obj['relationships']:
        newRel = []
        if (sg_obj['objects'][rel[0]] in sg_obj_list) \
          and (sg_obj['objects'][rel[2]] in sg_obj_list):
          preds.add(rel[1])
          s_idx = rel[0]
          newRel.append(idx_map[s_idx])
          newRel.append(rel[1])
          o_idx = rel[2]
          newRel.append(idx_map[o_idx])      
          self.image_id_to_relationships[image_caption_id].append(newRel)
      pred_counter.update(preds)

    object_names = ['__image__']
    for name, count in object_name_counter.most_common():
      object_names.append(name)
    print('Found %d object categories.' % (len(object_names)))
    
    # Prune images that have too few or too many objects
    new_image_ids = []
    total_objs = 0
    for image_id in self.image_ids:
      num_objs = len(self.image_id_to_sg_objects[image_id])
      total_objs += num_objs
      if min_objects_per_image <= num_objs <= max_objects_per_image:
        new_image_ids.append(image_id)
    self.image_ids = new_image_ids
    print('After pruning, %d images left.' % (len(self.image_ids)))

    error_imgs_file = "error_imgids.txt"
    self.image_ids = read_error_img_ids(error_imgs_file)
    print('Read %d error images ids.' % (len(self.image_ids)))

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

    self.vocab['pred_name_to_idx'] = pred_name_to_idx
    self.vocab['pred_idx_to_name'] = pred_idx_to_name

  def match_objs(self, sg_objs, coco_objs):
    match = {}
    idx_map = {}
    sg_idx = 0
    new_sg_idx = 0
    for sg_obj in sg_objs:
      for coco_obj in coco_objs:
        if ((sg_obj in coco_obj) or (coco_obj in sg_obj)):
          match[sg_obj] = coco_obj
      if match.get(sg_obj, None) != None:
        idx_map[sg_idx] = new_sg_idx
        new_sg_idx += 1
      sg_idx += 1
      
    return match, idx_map 

  def find_available_caption_id(self, image_id):
    # find the available image_caption_id
    for suf in self.suffix:
      new_caption_id = str(image_id) + suf
      if new_caption_id not in self.caption_id_map[image_id]:
        self.caption_id_map[image_id].append(new_caption_id)
        return new_caption_id

  def set_image_size(self, image_size):
    print('called set_image_size', image_size)
    transform = [Resize(image_size), T.ToTensor()]
    if self.normalize_images:
      transform.append(imagenet_preprocess())
    self.transform = T.Compose(transform)
    self.image_size = image_size

  def total_objects(self):
    total_objs = 0
    for i, image_id in enumerate(self.image_ids):
      if self.max_samples and i >= self.max_samples:
        break
      num_objs = len(self.image_id_to_objects[image_id])
      total_objs += num_objs
    return total_objs

  def __len__(self):
    if self.max_samples is None:
      return len(self.image_ids)
    return min(len(self.image_ids), self.max_samples)

  def __getitem__(self, index):
    """
    Get the pixels of an image, and a random synthetic scene graph for that
    image constructed on-the-fly from its COCO object annotations. We assume
    that the image will have height H, width W, C channels; there will be O
    object annotations, each of which will have both a bounding box and a
    segmentation mask of shape (M, M). There will be T triples in the scene
    graph.

    Returns a tuple of:
    - image: FloatTensor of shape (C, H, W)
    - objs: LongTensor of shape (O,)
    - boxes: FloatTensor of shape (O, 4) giving boxes for objects in
      (x0, y0, x1, y1) format, in a [0, 1] coordinate system
    - masks: LongTensor of shape (O, M, M) giving segmentation masks for
      objects, where 0 is background and 1 is object.
    - triples: LongTensor of shape (T, 3) where triples[t] = [i, p, j]
      means that (objs[i], p, objs[j]) is a triple.
    """
    image_id = self.image_ids[index]
    print(str(image_id))
    
    filename = self.image_id_to_filename[image_id]
    image_path = os.path.join(self.image_dir, filename)
    with open(image_path, 'rb') as f:
      with PIL.Image.open(f) as image:
        WW, HH = image.size
        image = self.transform(image.convert('RGB'))

    H, W = self.image_size
    objs, boxes, masks = [], [], []
    for object_name in self.image_id_to_sg_objects[image_id]:
      objs.append(self.vocab['object_name_to_idx'][object_name])
      # bbox
      x, y, w, h = self.image_id_to_objects_bbox[image_id][object_name]
      x0 = x / WW
      y0 = y / HH
      x1 = (x + w) / WW
      y1 = (y + h) / HH
      boxes.append(torch.FloatTensor([x0, y0, x1, y1]))
      
      # This will give a numpy array of shape (HH, WW)
      mask = seg_to_mask(self.image_id_to_objects_seg[image_id][object_name], WW, HH)

      # Crop the mask according to the bounding box, being careful to
      # ensure that we don't crop a zero-area region
      mx0, mx1 = int(round(x)), int(round(x + w))
      my0, my1 = int(round(y)), int(round(y + h))
      mx1 = max(mx0 + 1, mx1)
      my1 = max(my0 + 1, my1)
      mask = mask[my0:my1, mx0:mx1]
      mask = imresize(255.0 * mask, (self.mask_size, self.mask_size),
                      mode='constant')
      mask = torch.from_numpy((mask > 128).astype(np.int64))
      masks.append(mask)
    
    assert len(masks) == len(objs)
    assert len(boxes) == len(objs)
    # Add dummy __image__ object
    objs.append(self.vocab['object_name_to_idx']['__image__'])
    boxes.append(torch.FloatTensor([0, 0, 1, 1]))
    masks.append(torch.ones(self.mask_size, self.mask_size).long())

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
    __image__ = self.vocab['object_name_to_idx']['__image__']
    real_objs = []
    if num_objs > 1:
      real_objs = (objs != __image__).nonzero().squeeze(1)
    for cur in real_objs:
      choices = [obj for obj in real_objs if obj != cur]
      if len(choices) == 0 or not self.include_relationships:
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
      p = self.vocab['pred_name_to_idx'][p]
      triples.append([s, p, o])

    for rel in self.image_id_to_relationships[image_id]:
      s = int(rel[0])
      p = self.vocab['pred_name_to_idx'].get(rel[1], None)
      o = int(rel[2])
      if s is not None and o is not None and p is not None:
        triples.append([s, p, o])
    
    # Add __in_image__ triples
    O = objs.size(0)
    in_image = self.vocab['pred_name_to_idx']['__in_image__']
    for i in range(O - 1):
      triples.append([i, in_image, O - 1])
    
    triples = torch.LongTensor(triples)
    return image, objs, boxes, masks, triples
    

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


def coco_sg_collate_fn(batch):
  """
  Collate function to be used when wrapping CocoSceneGraphDataset in a
  DataLoader. Returns a tuple of the following:

  - imgs: FloatTensor of shape (N, C, H, W)
  - objs: LongTensor of shape (O,) giving object categories
  - boxes: FloatTensor of shape (O, 4)
  - masks: FloatTensor of shape (O, M, M)
  - triples: LongTensor of shape (T, 3) giving triples
  - obj_to_img: LongTensor of shape (O,) mapping objects to images
  - triple_to_img: LongTensor of shape (T,) mapping triples to images
  """
  all_imgs, all_objs, all_boxes, all_masks, all_triples = [], [], [], [], []
  all_obj_to_img, all_triple_to_img = [], []
  obj_offset = 0
  for i, (img, objs, boxes, masks, triples) in enumerate(batch):
    all_imgs.append(img[None])
    if objs.dim() == 0 or triples.dim() == 0:
      continue
    O, T = objs.size(0), triples.size(0)
    all_objs.append(objs)
    all_boxes.append(boxes)
    all_masks.append(masks)
    triples = triples.clone()
    triples[:, 0] += obj_offset
    triples[:, 2] += obj_offset
    all_triples.append(triples)

    all_obj_to_img.append(torch.LongTensor(O).fill_(i))
    all_triple_to_img.append(torch.LongTensor(T).fill_(i))
    obj_offset += O

  all_imgs = torch.cat(all_imgs)
  all_objs = torch.cat(all_objs)
  all_boxes = torch.cat(all_boxes)
  all_masks = torch.cat(all_masks)
  all_triples = torch.cat(all_triples)
  all_obj_to_img = torch.cat(all_obj_to_img)
  all_triple_to_img = torch.cat(all_triple_to_img)

  out = (all_imgs, all_objs, all_boxes, all_masks, all_triples,
         all_obj_to_img, all_triple_to_img)
  return out

def read_error_img_ids(error_imgs_file):
  with open(error_imgs_file, 'r') as f:
      img_ids = f.readlines() 
  
  new_img_ids = []
  for id in img_ids:
    len_id = len(id)
    if (len_id > 2):
      new_img_ids.append(id[0:len_id//2])
  
  return new_img_ids

