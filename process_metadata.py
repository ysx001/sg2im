import argparse, json, os, glob
from collections import Counter, defaultdict


COCO_DIR = 'datasets/coco/annotations/'

parser = argparse.ArgumentParser()
parser.add_argument('--instance_json', \
    default=os.path.join(COCO_DIR, 'instances_train2017.json'))
parser.add_argument('--sg_json', \
    default=os.path.join(COCO_DIR, 'sg_train2017.json'))

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

def match(sg_objs, coco_objs):
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

if __name__ == '__main__':
  args = parser.parse_args()
#   read_coco(args.instance_json, args.sg_json)
#   read_json(args.instance_json)
#   combine_json()
  a = ['aaa', 'aaa', 'b', 'c', 'd']
  b = ['aaa', 'c', 'd']
  match, idx_map = match(a, b)
  print(match)
  print(idx_map)