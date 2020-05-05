# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 18:06:37 2020

@author: Raaj
"""

import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary
import pandas as pd
import glob
import random



class MsvdDataset(data.Dataset):
    """MSVD Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self,transform=None):
        
        self.transform = transform
        
        
    def getAll(self):
            transform = transforms.Compose([ 
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(), 
                    transforms.ToTensor(), 
                    transforms.Normalize((0.485, 0.456, 0.406), 
                                         (0.229, 0.224, 0.225))])
                    
            os.chdir(r'./msvd dataset/YouTubeClips/')
            vidFiles = glob.glob('*.avi')
            df = pd.read_csv('E:/jupyterNotebook/our_project/msvd dataset/MSR Video Description Corpus.csv')
            df = df[df['Language'] == 'English']
            #print(len(vidFiles))
            
            random.shuffle(vidFiles)
            data_loader = []
            for i,item in enumerate(vidFiles):
                if(i==100):
                    break
                name_with_duration = item[:-4]
                
                os.chdir(r'E:/jupyterNotebook/our_project/data/resized frames/')
                images_for_item = glob.glob(name_with_duration+'*')
                root = 'E:/jupyterNotebook/our_project/data/resized frames/'
                stacked_images = []
                for frame in images_for_item:
                    image = Image.open(os.path.join(root, frame)).convert('RGB')
                    if transform is not None:
                        image = transform(image)
                    stacked_images.append(image)
                    
                stacked_images = torch.stack(stacked_images, 0)
                
                
                splited_name = name_with_duration.split('_')
                start = splited_name[-2]
                end = splited_name[-1]
                extention_len = len(start+end)+2
                only_vid_name = name_with_duration[:-extention_len]
                captions = df[df['VideoID'] == only_vid_name]
                captions = captions[captions['Start'] == int(start)]
                captions = captions['Description']
                if len(captions) <1:
                    continue
                first_caption = captions.iloc[0]
                print(name_with_duration)
                
                #print(i,name_with_duration,only_vid_name,len(captions))
                
                # Load vocabulary wrapper
                with open('E:/jupyterNotebook/our_project/data/vocab.pkl', 'rb') as f:
                    vocab = pickle.load(f)
                    
                target_list = []
                length_list = []
                for item in captions:
                    # Convert captions (string) to word ids.
                    tokens = nltk.tokenize.word_tokenize(str(item).lower())
                    #print(tokens)
                    caption = []
                    caption.append(vocab('<start>'))
                    caption.extend([vocab(token) for token in tokens])
                    caption.append(vocab('<end>'))
                    target = torch.Tensor(caption)
                    
                    length = torch.tensor([len(caption)])
                    
                    target_list.append(target)
                    length_list.append(length)
                
                data_loader.append([stacked_images,target_list,length_list])
                #print(i,stacked_images.shape,target.shape,length.shape) 
                #print(len(target_list),len(length_list))
                #print(length)
            #print(len(data_loader[0][1]))
            return data_loader    
                
                
                
#a = MsvdDataset()
#cap = a.getAll()           
