# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 21:45:43 2020

@author: Raaj
"""
import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import data_loader
from model import EncoderCNN, DecoderRNN
from build_vocab import Vocabulary
from torch.nn.utils.rnn import pack_padded_sequence

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
        
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
        
    # Build the models
    encoder = EncoderCNN(args.embed_size).to(device)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters())+list(encoder.linear.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    
    obj = data_loader.MsvdDataset()     
    datas = obj.getAll()
    #print(len(datas))
    os.chdir(r'E:/jupyterNotebook/our_project/')
    # Train the models
    total_step = len(datas)
    for epoch in range(args.num_epochs):
        for i, (images, captions, lengths) in enumerate(datas):
            
            
            #print(epoch,i,images.shape)
            # Set mini-batch dataset
            images = images.to(device)

            # Forward, backward and optimize
            features = encoder(images)
            features = features.cpu().detach().numpy()
            features = features.mean(axis=0)
            features = torch.from_numpy(features).view(1,-1).to(device)
            #print(features.shape)
            
            for j in range(1):
            #for j in range(len(captions)):
                captions[j] = captions[j].long()
                captions[j] = captions[j].view(1,-1).to(device)
                targets = pack_padded_sequence(captions[j], lengths[j], batch_first=True)[0]
                
                outputs = decoder(features, captions[j], lengths[j])
                #print(targets.shape)
                #print(outputs.shape)
                loss = criterion(outputs, targets)
                decoder.zero_grad()
                #encoder.zero_grad()
                loss.backward()
                optimizer.step()
            
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch+1, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item()))) 

            #print(os.path)
            if (i+1) % 25==0:#args.save_step == 0:
                torch.save(decoder.state_dict(), os.path.join(
                    'E:\jupyterNotebook\our_project\models', 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
                torch.save(encoder.state_dict(), os.path.join(
                    'E:\jupyterNotebook\our_project\models', 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))
'''
            # Print log info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item()))) 
                
            # Save the model checkpoints
            if (i+1) % args.save_step == 0:
                torch.save(decoder.state_dict(), os.path.join(
                    args.model_path, 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
                torch.save(encoder.state_dict(), os.path.join(
                    args.model_path, 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/models/' , help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    #parser.add_argument('--image_dir', type=str, default='data/resized frames/', help='directory for resized images')
    #parser.add_argument('--caption_path', type=str, default='data/annotations/captions_train2014.json', help='path for train annotation json file')
    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=100, help='step size for saving trained models')
    
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)