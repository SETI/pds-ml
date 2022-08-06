# Simple utilities for reading and manipulating images

import os

def get_images(image_path):

    '''
    Get list of images in a folder with subfolders.

    Args:
        image_path (str) : Path to ImageFolder Dataset
    
    Returns:
        (str list) : list of images in a folder
    '''

    ims = []
    for folder in os.listdir(image_path):
        for im in os.listdir(f'{image_path}/{folder}'):
            ims.append(f'{image_path}/{folder}/{im}')

    return ims
    
