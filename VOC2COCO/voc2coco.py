#encoding:utf-8

import json
import os, sys
import xml.etree.ElementTree as ET
import argparse
from collections import OrderedDict
from tqdm import tqdm


def get_elements(root, childElementName):
    elements = root.findall(childElementName)
    return elements


def get_element(root, childElementName):
    element = root.find(childElementName)
    return element

def summary_classes(xml_dir):
    xml_list = sorted(os.listdir(xml_dir))
    classes = set()

    for xml in xml_list:
        f = os.path.join(xml_dir, xml)
        tree = ET.parse(f)
        root = tree.getroot()
        for bbox in root.findall('object'):
            classes.add(bbox.find('name').text)
    
    classes = sorted(list(classes))
    return classes

def voc2coco(xml_dir,output):
    
    voc_xmls_list = sorted(os.listdir(xml_dir))
    image_id = 0
    bbox_id = 0

    coco_json = {"images":[], "annotations":[], "categories":[]}

    for key,val in PRE_DEFINE_CATEGORIES.items():
        temp_dict = {}
        temp_dict['id'] = int(val)
        temp_dict['name'] = key
        temp_dict['supercategory'] = 'object'
        coco_json['categories'].append(temp_dict)
    
    for xml_fileName in tqdm(voc_xmls_list):

        xml_fullName = xml_fileName
        tree = ET.parse(os.path.join(xml_dir,xml_fullName)) 
        root = tree.getroot()         
        
        #image: file_name
        filename = get_element(root, 'filename').text 
        #image: id
        image_id = image_id + 1
        
        size = get_element(root, 'size')
        #image: width
        img_width = int(get_element(size, 'width').text)
        #image: height
        img_height = int(get_element(size, 'height').text)

        image = {
            'file_name': filename, 
            'id':image_id,
            'width': img_width,
            'height': img_height
            }

        coco_json['images'].append(image)


        for obj in get_elements(root, 'object'):
            # annotation: category_id
            category = get_element(obj, 'name').text
            if category not in PRE_DEFINE_CATEGORIES:
                new_id = len(PRE_DEFINE_CATEGORIES) + 1
                PRE_DEFINE_CATEGORIES[category] = new_id
            category_id = PRE_DEFINE_CATEGORIES[category]

            # annotation: id
            bbox_id += 1

            # annotation: bbox
            bndbox = get_element(obj, 'bndbox')
            xmin = int(get_element(bndbox, 'xmin').text)
            ymin = int(get_element(bndbox, 'ymin').text)
            xmax = int(get_element(bndbox, 'xmax').text)
            ymax = int(get_element(bndbox, 'ymax').text)
            try:
                assert(xmax > xmin)
                assert(ymax > ymin)
            except:
                print('\nError Annotation:',xml_fileName,', Corrupt bbox ignored !')
                continue
            bbox_width = abs(xmax - xmin)
            bbox_height = abs(ymax - ymin)

            annotation = {
                'id': bbox_id,
                'image_id': image_id,
                'category_id': category_id,
                'segmentation': [],
                'area': bbox_width * bbox_height, 
                'bbox':[xmin, ymin, bbox_width, bbox_height],
                'iscrowd': 0
                }

            coco_json['annotations'].append(annotation)


    print("Num of categories: %s" % len(coco_json['categories']))
    print("Num of images: %s" % len(coco_json['images']))
    print("Num of annotations: %s" % len(coco_json['annotations']))
    print("LabelMap:", PRE_DEFINE_CATEGORIES)
    print ("Inspect Json['categories']:",coco_json['categories'])
    
    # write json
    with open(output, 'w') as outfile:  
        outfile.write(json.dumps(coco_json))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="voc2coco")
    parser.add_argument('--labelmap',type = str ,help="path to customized labelmap.txt",default="")
    parser.add_argument('--xml_dir',type = str ,help = 'directory to xml files',default="./Annotations")
    parser.add_argument('--output',type = str ,help = 'path to output json files',default="./data.json")

    args = parser.parse_args()

    if args.labelmap:
        PRE_DEFINE_CATEGORIES = OrderedDict()
        with open(args.labelmap,'r') as f:
            lines = [x.strip() for x in f.readlines()]
            for line in lines:
                val, key = line.split(',')
                PRE_DEFINE_CATEGORIES[key] = int(val)
    else:
        classes = summary_classes(args.xml_dir)
        ids = [x for x in range(1,len(classes)+1)]
        PRE_DEFINE_CATEGORIES = OrderedDict(zip(classes,ids))


    print('start convert')
    voc2coco(args.xml_dir,args.output)
    print('\nconvert finished!')