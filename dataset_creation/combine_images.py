# -*- coding: utf-8 -*-
"""
Created on Fri May 12 15:52:09 2023

@author: Shaik
"""
#importing necessary libraries
import random
import torch
from PIL import Image

""" The combine_images function accepts two image inputs: an object image and a background image"""
def combine_images(object_image, background_image):
    
    # get width, height of both the images i.e background images and object images
    background_width, background_height = background_image.size
    object_width, object_height = object_image.size
    
    #calculate 
    max_x_offset = background_width - object_width
    max_y_offset = background_height - object_height
    
    x_offset = random.randint(0, max_x_offset)
    y_offset = random.randint(0, max_y_offset)
    xmin = x_offset
    ymin = y_offset
    xmax = x_offset + object_width
    ymax = y_offset + object_height
    
    
    # create a new image with the same size and mode as the ''L''
    alpha_channel = Image.new('L', object_image.size, 0)
    alpha_channel.paste(object_image, None )
    alpha_channel = alpha_channel.point(lambda x: 255 if x != 255 else 0)
    object_image.putalpha(alpha_channel)
    
     
    background_image.paste(object_image, (x_offset, y_offset), object_image)
    
    # Create target dictionary with transformed bounding box coordinates
    target = {}
    target['width'] = object_width
    target['height'] = object_height
    target['bnbbox'] = torch.tensor([[xmin,xmax, ymin,ymax]], dtype=torch.float32)
    target['depth'] = 24
    # target['area'] = torch.tensor([(xmax-xmin) * (ymax-ymin)], dtype=torch.float32)
    
    return background_image, target 
