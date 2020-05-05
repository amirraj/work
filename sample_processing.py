# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 14:11:42 2020

@author: Aspire
"""

import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
from torchvision import transforms 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from PIL import Image
import glob
import cv2


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def resize_image(image, size):
    """Resize an image to the given size."""
    return image.resize(size, Image.ANTIALIAS)

def resize_images(image_dir='./sample/frames/', output_dir='./sample/resized frames', size=[256,256]):
    """Resize the images in 'image_dir' and save into 'output_dir'."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = os.listdir(image_dir)
    #num_images = len(images)
    for i, image in enumerate(images):
        with open(os.path.join(image_dir, image), 'r+b') as f:
            with Image.open(f) as img:
                img = resize_image(img, size)
                img.save(os.path.join(output_dir, image), img.format)
        
def getFrame(sec,path,vidcap,count):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    foo = os.path.basename(path)
    foo = foo[:-4]
    dest_path = "./sample/frames/"+ "sample_"
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite(dest_path+str(count)+".jpg", image)# save frame as JPG file
    return hasFrames

def load_video(video_path='./sample/sample.avi', transform=None):
    
    videos = glob.glob('./sample' + '/*.avi')
    for i,item in enumerate(videos):
        vidcap = cv2.VideoCapture(item)
        sec = 0
        frameRate = 0.33  #//it will capture image in each 0.33 second
        count=1
        success = getFrame(sec,item,vidcap,count)
        while success:
            count = count + 1
            sec = sec + frameRate
            sec = round(sec, 2)
            success = getFrame(sec,item,vidcap,count)
            
    resize_images()
    
    
    transform = transforms.Compose([ 
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(), 
                    transforms.ToTensor(), 
                    transforms.Normalize((0.485, 0.456, 0.406), 
                                         (0.229, 0.224, 0.225))])
    
    name = 'sample_'
    os.chdir(r'E:/jupyterNotebook/our_project/sample/resized frames/')
    images_for_item = glob.glob(name+'*')
    root = 'E:/jupyterNotebook/our_project/sample/resized frames/'
    stacked_images = []
    for frame in images_for_item:
        image = Image.open(os.path.join(root, frame)).convert('RGB')
        if transform is not None:
            image = transform(image)
        stacked_images.append(image)
        
    stacked_images = torch.stack(stacked_images, 0)
    #print(stacked_images.shape)
    
    return stacked_images
#load_video()