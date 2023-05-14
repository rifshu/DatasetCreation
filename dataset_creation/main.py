
#import necessary librabries
import os
import random
import torch
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
import sys,  inspect
from torchvision import transforms
import xml.etree.ElementTree as ET

"""combine_images: function used to generate final combines images and target
    save_target_as_xml : function used to save the targets in the xml format """
from combine_images import combine_images
from xml_target import save_target_as_xml


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

grandparentdir = os.path.dirname(parentdir)
sys.path.insert(0,grandparentdir)

grandgrandparentdir = os.path.dirname(grandparentdir)
sys.path.insert(0,grandgrandparentdir)


class CustomDataset(Dataset):
    def __init__(self, object_dir, background_dir, target_dir, transform=None, num_samples = None):
        self.object_dir = object_dir
        self.background_dir = background_dir
        self.target_dir = target_dir
        self.transform = transform
        
        self.object_filenames = sorted([f for f in os.listdir(self.object_dir) if f.endswith('.png')])
        
        # self.object_filenames =  100000
        
        self.background_filenames = sorted([f for f in os.listdir(self.background_dir) if f.endswith('.jpg') or f.endswith('.jpeg')])
        if num_samples is None:
            num_samples = len(self.object_filenames) * len(self.background_filenames)
        else:
            num_samples = min(num_samples, len(self.object_filenames) * len(self.background_filenames))

        self.num_samples = num_samples

    def __len__(self):
        return (self.num_samples)
    
    def __getitem__(self, idx):
        
        
        
        # Calculate indices for object and background images
        num_objects = len(self.object_filenames)
        background_idx = idx % len(self.background_filenames)
        object_idx = (idx // len(self.background_filenames)) % num_objects
        # object_idx = idx % num_objects
        
        # Load object and background images, and apply any transforms
        object_filename = self.object_filenames[object_idx]
        
        # Extract class label from filename using regular expression pattern
        # pattern = r'([a-zA-Z_]+)\.png'
        # match = re.match(pattern, object_filename)
        class_label =object_filename.split('.')[0]
        
        # Load object and background images, and apply any transforms
        object_image = Image.open(os.path.join(self.object_dir, object_filename)).convert('RGB')
        background_filename = random.choice([f for f in os.listdir(self.background_dir) if f.endswith('.jpg') or f.endswith('.jpeg')])
        background_image = Image.open(os.path.join(self.background_dir, background_filename)).convert('RGB')
        
        """transform can be applied to both background and object images respectively.
            help in increment of number of samples(combined image) and final targets"""
        if self.transform:
            object_image = self.transform(object_image)
            # background_image = self.transform(background_image)
        
        # Combine object and background images   
        object_image = transforms.ToPILImage()(object_image)
        # background_image = transforms.ToPILImage()(background_image)
        final_image, target = combine_images(object_image, background_image)
        target['labels'] = class_label
        
        """based on requirement target['image_id']  , target['iscrowd'] can also be calculated."""
        # target['image_id'] = torch.tensor([idx])
        # target['iscrowd'] = torch.zeros((1,), dtype=torch.int64)
        
        # Save image and target as XML files in a same target dir
        image_filename = f'image_{idx}.jpg'
        target_filename = f'image_{idx}.xml'
        final_image.save(os.path.join(self.target_dir, image_filename))
        save_target_as_xml(target, target_filename , os.path.join(self.target_dir, target_filename))
                
        return 
    





if __name__ == '__main__':
    
    custom_dataset = CustomDataset(object_dir='C:/Git_hub/DatasetCreation/dataset_creation/raw_data/objects', 
                                   background_dir='C:/Git_hub/DatasetCreation/dataset_creation/raw_data/backgrounds', 
                                   target_dir= 'C:/Git_hub/DatasetCreation/dataset_creation/raw_data/combined_dataset',
                                   transform=transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()])
                                   , num_samples = 150000)
    
    
    for idx in tqdm(range(len(custom_dataset)), desc= 'Generating_Data'):
       Datasett = custom_dataset[idx]
        
