# -*- coding: utf-8 -*-
"""
Created on Fri May 12 15:51:24 2023

@author: Shaik
"""
import xml.etree.ElementTree as ET

def save_target_as_xml(target, filename ,target_path ):
    
    annotation = ET.Element('annotation')

    folder = ET.SubElement(annotation, 'folder')
    folder.text = 'my_dataset'

    filename_elem = ET.SubElement(annotation, 'filename')
    filename_elem.text = filename

    path = ET.SubElement(annotation, 'path')
    path.text = target_path

    source = ET.SubElement(annotation, 'source')
    database = ET.SubElement(source, 'database')
    database.text = 'Rifshu_custom_dataset'

    size = ET.SubElement(annotation, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(target['width'])
    height = ET.SubElement(size, 'height')
    height.text = str(target['height'])
    depth = ET.SubElement(size, 'depth')
    depth.text = str(target['depth'])

    # segmented = ET.SubElement(annotation, 'segmented')
    # segmented.text = '0'

    object_elem = ET.SubElement(annotation, 'object')
    name = ET.SubElement(object_elem, 'name')
    name.text = str(target['labels'])

    # pose = ET.SubElement(object_elem, 'pose')
    # pose.text = 'Unspecified'

    # truncated = ET.SubElement(object_elem, 'truncated')
    # truncated.text = '0'

    # difficult = ET.SubElement(object_elem, 'difficult')
    # difficult.text = '0'

    # occluded = ET.SubElement(object_elem, 'occluded')
    # occluded.text = '0'

    bndbox = ET.SubElement(object_elem, 'bndbox')
    
    xmin = ET.SubElement(bndbox, 'xmin')
    xmin.text = str(int(target['bnbbox'][0][0]))
    
    xmax = ET.SubElement(bndbox, 'xmax')
    xmax.text = str(int(target['bnbbox'][0][1]))
    
    ymin = ET.SubElement(bndbox, 'ymin')
    ymin.text = str(int(target['bnbbox'][0][2]))
    
    ymax = ET.SubElement(bndbox, 'ymax')
    ymax.text = str(int(target['bnbbox'][0][3]))

    tree = ET.ElementTree(annotation)
    tree.write(target_path)
    
    
    
    
        
#     def collate_fn(self, batch):
#         images = []
#         targets = []
#         for img, target in batch:
#             images.append(torch.tensor(img).unsqueeze(0))
#             targets.append(target)
#         return torch.cat(images), targets
    