# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 15:27:31 2020

@author: Raaj
"""
import argparse
import os
from PIL import Image
import cv2
import glob



def getFrame(sec,path,vidcap,count):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    foo = os.path.basename(path)
    foo = foo[:-4]
    dest_path = "./data/frames/"+path[28:-4] + "_"
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite(dest_path+str(count)+".jpg", image)# save frame as JPG file
    return hasFrames

def split():
    videos = glob.glob('./msvd dataset/YouTubeClips' + '/*.avi')
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
        if i % 10 == 0:
            print("[{}/{}] is done".format(i+1, len(videos)))
 

def resize_image(image, size):
    """Resize an image to the given size."""
    return image.resize(size, Image.ANTIALIAS)

def resize_images(image_dir, output_dir, size):
    """Resize the images in 'image_dir' and save into 'output_dir'."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = os.listdir(image_dir)
    num_images = len(images)
    for i, image in enumerate(images):
        with open(os.path.join(image_dir, image), 'r+b') as f:
            with Image.open(f) as img:
                img = resize_image(img, size)
                img.save(os.path.join(output_dir, image), img.format)
        if (i+1) % 100 == 0:
            print ("[{}/{}] Resized the images and saved into '{}'."
                   .format(i+1, num_images, output_dir))

def main(args):
    split()
    image_dir = args.image_dir
    output_dir = args.output_dir
    image_size = [args.image_size, args.image_size]
    resize_images(image_dir, output_dir, image_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='./data/frames/',
                        help='directory for train images')
    parser.add_argument('--output_dir', type=str, default='./data/resized frames/',
                        help='directory for saving resized images')
    parser.add_argument('--image_size', type=int, default=256,
                        help='size for image after processing')
    args = parser.parse_args()
    main(args)           
