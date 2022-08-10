# Simple utilities for reading and manipulating images

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def get_images(image_path):

    '''
    Get list of images in a folder with subfolders.

    Args:
        image_path (str) : Path to ImageFolder Dataset
    
    Returns:
        (str list) : list of images in a folder
    '''

   #ims = []
   #for folder in os.listdir(image_path):
   #    for im in os.listdir(f'{image_path}/{folder}'):
   #        ims.append(f'{image_path}/{folder}/{im}')
        
    # TODO: get glob to not list directories without needing the '.*'
    imageFiles = glob.glob(image_path+'/**/*.*', recursive=True)

    return imageFiles

def plot_images(imageFiles, title_text='', values=None):
    """
    Plots the images in the list <imageFiles> in a 4xn grid pattern

    Parameters
    ----------
    imageFiles : str list
        The list of image filenames in the order of classification
    title_text : str
        A title to give each image, printed after "Rank = #;"
    values : int list(len(imageFiles))
        A value to print in the title after title_text for each image
    """

    n_to_plot = len(imageFiles)

    if values is not None:
        assert len(values) == n_to_plot, '<values> must be same length as <imageFiles>'

    ncol = 4
    nrow = int(np.ceil(n_to_plot / ncol))

    # Make sure column width is multiple of ncol
    colWidth = ncol * np.ceil(n_to_plot / ncol)
   #fig = plt.figure(figsize=(30,colWidth*2))
    fig = plt.figure(figsize=(20,colWidth*1.5))
    subplots = []

    for i in range(n_to_plot):
        # Plot the images
        subplots.append(fig.add_subplot(nrow,ncol,i+1))
        subplots[-1].set_axis_off()
        if values is not None:
            subplots[-1].set_title(f"Rank = {i}; {title_text} = {values[i]}",size=10)
        else:
            subplots[-1].set_title(f"Rank = {i}; {title_text}",size=10)
        f = imageFiles[i]
        im = Image.open(f).convert('RGB')
        subplots[-1].imshow(im)

    fig.show()
    
