# This module is used to apply a Self-Supervised Learning model to a set of data to evaulate the scores.

import torch
import torchvision.transforms as transforms
import glob
from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import shutil

from .SSL import supported_techniques
from ..utilities.image_utils import get_images

def evaluate_model(model_path, technique, image_path, gpu_index=0):

    '''      
    Generate Embeddings or classification for a given folder of images and a stored model.

    Parameters
    ----------
    model_path : str
        Path to PyTorch Lightning Model
    technique : 
        model technique used {SIMCLR, SIMSIAM or CLASSIFIER}
        See: SSL.supported_techniques
    image_path : str
        Path to ImageFolder Dataset
    gpu_index : int
        Specify the GPU index to use for processing
        Only useful for a multi-GPU machine
    
    Returns
    -------
    evaluation : (torch.Tensor) 
        Embeddings of the given model on the specified dataset.
        If evaluating a classifer then returns classification scores
    imageFiles : str list
        The list of image filenames in the order evaluated
    '''

    # Used the correct loader for the specified technique
    techniqueClass = supported_techniques[technique]

    gpu_index = 0

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print('Using CUDA')
       #model = torch.load(model_path)

        model = techniqueClass.load_from_checkpoint(model_path)
    else: 
        print('Using CPU')
        model = torch.load(model_path, map_location = device)

    image_size = model.image_size
    embedding_size = model.encoder.embedding_size

    # Permute image so that it is the correct orientation for our model
    def to_tensor(pil):
        return torch.tensor(np.array(pil)).permute(2,0,1).float()

    t = transforms.Compose([
                transforms.Resize((image_size,image_size)),
                transforms.Lambda(to_tensor)
                ])

    model.eval()
    if device == 'cuda':
        model.cuda(gpu_index)


    #***
    # Extract embeddings for all the images
    imageFiles = get_images(image_path)
    if technique == 'CLASSIFIER':
        # The output has only two columns
        evaluation = np.zeros([len(imageFiles), 2])
    else:
        evaluation = np.zeros([len(imageFiles), embedding_size])
    for idx in tqdm(range(len(imageFiles)), 'Computing model image embeddings'):
        f = imageFiles[idx]
        
        im = Image.open(f).convert('RGB')
        
        datapoint = t(im).unsqueeze(0).cuda(gpu_index) #only a single datapoint so we unsqueeze to add a dimension
        
        with torch.no_grad():
            evaluation[idx,:] = np.array(model(datapoint)[0].cpu()) #get evaluation


    return evaluation, imageFiles

#*************************************************************************************************************
def plot_top_n_images(classification, imageFiles,  top_n=100, top_n_copy_loc=None):
    """
    Displays the top N images based on the classification score

    Parameters
    ----------
    classification : float array[nImages, 2]
        The classification scores from evaluate_model
    imageFiles : str list
        The list of image filenames in the order of classification
    top_n_copy_loc : str
        Path to directory to copy the top n figures to

    """

    # Confirm a classification matrix is same size as list of images
    assert len(classification[:,0]) == len(imageFiles), \
        'Classification and imageFile smust be the same length'

    # Sort the images by classification score
    sorted_images_idx = np.argsort(classification[:,0])[::-1]


    ncol = 4
    nrow = int(np.ceil(top_n / ncol))

    # Make sure column width is multiple of ncol
    colWidth = ncol * np.ceil(top_n / ncol)
    fig = plt.figure(figsize=(30,colWidth*2))
    subplots = []

    for i in range(top_n):
        # Copy image to directory
        if top_n_copy_loc is not None:
            shutil.copy(imageFiles[sorted_images_idx[i]], top_n_copy_loc)
        
        # Plot the images
        subplots.append(fig.add_subplot(nrow,ncol,i+1))
        subplots[-1].set_axis_off()
        subplots[-1].set_title(f"Rank = {i}; Score = {classification[sorted_images_idx[i],0]}",size=13)
        f = imageFiles[sorted_images_idx[i]]
        im = Image.open(f).convert('RGB')
        subplots[-1].imshow(im)

    if top_n_copy_loc is not None:
        print('Top images copied to {}'.format(top_n_copy_loc))
