import torch
import yaml
import os
import numpy as np


def xywh2xywhnc(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_boxes(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = ((x[..., 0] + (x[..., 2]) / 2)) / w  # x center
    y[..., 1] = ((x[..., 1] + (x[..., 3]) / 2)) / h  # y center
    y[..., 2] = (x[..., 2]) / w  # width
    y[..., 3] = (x[..., 3]) / h  # height
    return y

def parse_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    classes = {class_name: str(class_id) for class_id, class_name in data['names'].items()}
    return classes

def append_label_to_file(file_path, line):
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            file.write(line + '\n')
    else:
        with open(file_path, 'a') as file:
            file.write(line + '\n')


dataset_path = "../../../datasets"
annotations_train_path = "../../../datasets/annotations/instances_train2017.json"
labels_path = "../../../datasets/coco128/labels/train2017/"
yaml_file_path = '../coco128.yaml'  # Replace with the actual path to your YAML file

if not os.path.exists(labels_path):
    import json
    
    print("Parsing Labels")
    
    classes = parse_yaml(yaml_file_path)
    
    os.makedirs(os.path.dirname(labels_path), exist_ok=True)
    
    images = {}
    annotations = {}
    categories = {}

    # Open and read the JSON file
    with open(annotations_train_path, "r") as json_file:
        data = json.load(json_file)
        
    # Adjust categories to have same ids as in coco128.yaml    
    for category in data['categories']:
        categories[str(category['id'])] = category['name']
        
            
    # Save images metadata in dict for use with annotations
    for image in data['images']:
        img_id = str(image['id']) 
        images[img_id] = {}
        img_dict = images[img_id]

        img_dict['file_name'] = image['file_name'].replace(".jpg",".txt")
        img_dict['width'] = image['width']
        img_dict['height'] = image['height']
        
    for annotation in data['annotations']:
        category_id = str(annotation['category_id'])
        category_name = categories[category_id]
        
        if category_name not in classes.keys():
            continue
            
        annotation_id = str(annotation['id'])
        image_id = str(annotation['image_id'])
        category_id = str(annotation['category_id'])
        width, height, filename = images[image_id]['width'], images[image_id]['height'], images[image_id]['file_name']
        bbox = np.array(annotation['bbox']) 
        class_id = classes[category_name]

        if image_id not in annotations.keys():
            annotations[image_id] = []

        annotations[image_id].append({
            "class_id":class_id,
            "bbox" : bbox,
            "width" : width,
            "height":height,
        })

        bbox_n = xywh2xywhnc(bbox, w=width, h=height)

        line = "{} {}".format(class_id,' '.join(map(str, bbox_n)))
        append_label_to_file("{}/{}".format(labels_path,filename),line)
else:
    print("Labels already exist")
     
        
     