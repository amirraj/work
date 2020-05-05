# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 23:08:11 2020

@author: Raaj
"""

import torch
import numpy as np 
import argparse
import pickle 
import sample_processing as sp
from torchvision import transforms 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from PIL import Image
import shutil



# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
   
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build models
    encoder = EncoderCNN(args.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))

    # Prepare an video
    video = sp.load_video(args.video)
    video_tensor = video.to(device)
    
    # Generate an caption from the video
    features = encoder(video_tensor)
    features = features.cpu().detach().numpy()
    features = features.mean(axis=0)
    features = torch.from_numpy(features).view(1,-1).to(device)
    
    sampled_ids = decoder.sample(features)
    sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
    
    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)
    
    # Print out the image and the generated caption
    print (sentence)
    #image = Image.open(args.image)
    #plt.imshow(np.asarray(image))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--image', type=str, required=True, help='input image for generating caption')
    parser.add_argument('--video', type=str, default='sample/sample.avi', help='input video for generating caption')
    parser.add_argument('--encoder_path', type=str, default='models/encoder-89-25.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='models/decoder-89-25.ckpt', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    args = parser.parse_args()
    main(args)
