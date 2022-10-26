# This module is used to apply a Self-Supervised Learning model to a set of data to evaulate the scores.

import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import shutil
import os

from .SSL import supported_techniques
from ..utilities.image_utils import get_images, plot_images

class Evaluator():
    """ Class to evaluate a SSL model or classifier

    """

    def __init__(self, model_path, technique, softmax_outputs=False, gpu_index=0):
        """
        Constructor

        The SSL classifier model does not softmax the outputs. The classifier returns a real number. We have the option
        to apply a softmax to the output to get the result in the range [0,1], however, that has no real impact other
        than renormalize the output values. If softmax_outputs = True then when you call evaluate_model a softmax is
        applied to the outputs. Note that the normalization is dependent on the array of datums you are evaluating the
        model on. TODO: set this up so that it uses a staticly set normalization factor.

        Parameters
        ----------
        model_path : str
            Path to PyTorch Lightning Model
        technique : 
            model technique used {SIMCLR, SIMSIAM or CLASSIFIER}
            See: SSL.supported_techniques
        softmax_outputs : bool
            If True then apply a torch.nn.Softmax to the evaluation outputs, but only if technique=='CLASSIFIER'
        gpu_index : int
            Specify the GPU index to use for processing
            Only useful for a multi-GPU machine

        """

        self.model_path = model_path
        self.technique = technique
        self.softmax_outputs = softmax_outputs
        self.gpu_index = gpu_index

        # Used the correct loader for the specified technique
        techniqueClass = supported_techniques[self.technique]
 
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            print('Using CUDA')
           #model = torch.load(model_path)
 
            self.model = techniqueClass.load_from_checkpoint(model_path)
        else: 
            print('Using CPU')
            self.model = torch.load(model_path, map_location = self.device)
 
        image_size = self.model.image_size
 
        # Permute image so that it is the correct orientation for our model
        def to_tensor(pil):
            return torch.tensor(np.array(pil)).permute(2,0,1).float()
 
        self.image_transformer = transforms.Compose([
                    transforms.Resize((image_size,image_size)),
                    transforms.Lambda(to_tensor)
                    ])
 
        self.model.eval()
        if self.device == 'cuda':
            self.model.cuda(self.gpu_index)


        return




    def evaluate_model(self, image_path, verbosity=True, limit_to_n_images=None):

        '''      
        Generate Embeddings or classification for a given folder of images and a stored model.
 
        Parameters
        ----------
        image_path : str
            Path to image folder dataset
            or list of individual files
        verbosity : bool
            If True then use tqdm to show progress
        limit_to_n_images : int
            If not None then limit the number of images evaluated to N
        
        Returns
        -------
        evaluation : (torch.Tensor) 
            Embeddings of the given model on the specified dataset.
            If evaluating a classifer then returns classification scores
        imageFiles : str np.array
            The list of image filenames in the order evaluated
        selected_images : [OPTIONAL] np.array(limit_to_n_images)
            If limiting to N images, this is the indices of the selected images in imageFiles
            Only returned if limit_to_n_images is not None:
        '''
 
        # Check if we have a list of files or a path to a folder
        if isinstance(image_path, list):
            imageFiles = image_path
        elif os.path.isfile(image_path):
            imageFiles = [image_path]
        elif os.path.isdir(image_path):
            # This is a path to a folder with images
            imageFiles = get_images(image_path)
        else:
            raise Exception('Error reading image_path')

        imageFiles = np.array(imageFiles)

        # Limit to N images
        if limit_to_n_images is not None:
            selected_images = np.random.choice(len(imageFiles), size=limit_to_n_images, replace=False)
            imageFiles = imageFiles[selected_images]       
 
        #***
        # Extract embeddings for all the images
        if self.technique == 'CLASSIFIER':
            # The output has only two columns
            evaluation = np.zeros([len(imageFiles), 2])
        else:
            evaluation = np.zeros([len(imageFiles), self.model.encoder.embedding_size])
 
        for idx in tqdm(range(len(imageFiles)), 'Computing model image embeddings', disable=not verbosity):
            f = imageFiles[idx]
            
            im = Image.open(f).convert('RGB')
            
            if self.device == 'cuda':
                # Only a single datapoint so we unsqueeze to add a dimension
                datapoint = self.image_transformer(im).unsqueeze(0).cuda(self.gpu_index)
            else:
                datapoint = self.image_transformer(im).unsqueeze(0)
            
            with torch.no_grad():
                # Get evaluation
                evaluation[idx,:] = np.array(self.model(datapoint)[0].cpu()) 

      # #***
      # TODO: Figure out how to use torch DataLoader with the way the images are loaded for Lightning bolts SIMCLR,
      # which uses the above 
      # # Try using torchvision data loader
      # def to_tensor(pil):
      #     return torch.tensor(np.array(pil)).permute(2,0,1).float()
      # image_transformer_test = transforms.Compose([
      #             transforms.Resize((self.model.image_size,self.model.image_size)),
      #             transforms.Lambda(to_tensor)
      #             ])
      # imageFiles_test = torchvision.datasets.ImageFolder(image_path, transform = image_transformer_test)
      # embedding = torch.empty(size = (0, self.model.encoder.embedding_size)).cuda(self.gpu_index)
      # bs = 64
      # if len(imageFiles_test) < bs:
      #     bs = 1
      # loader = torch.utils.data.DataLoader(imageFiles_test, batch_size = bs, shuffle = False)
      # for batch in tqdm(loader, 'Computing model image embeddings'):
      #             
      #     x = batch[0].cuda(self.gpu_index)
      #     
      #     with torch.no_grad():
      #         embedding = torch.vstack((embedding, self.model(x)))
 
 
        if self.softmax_outputs:
            softmax_fcn = torch.nn.Softmax(dim=1)
            evaluation = np.array(softmax_fcn(torch.tensor(evaluation)))

        if limit_to_n_images is not None:
            return evaluation, imageFiles, selected_images
        else:
            return evaluation, imageFiles

#*************************************************************************************************************
def plot_top_n_images(classification, imageFiles, top_n=100, class_index=0, top_n_copy_loc=None):
    """
    Displays the top N images based on the classification score

    Parameters
    ----------
    classification : float array[nImages, 2]
        The classification scores from evaluate_model
    imageFiles : str list
        The list of image filenames in the order of classification
    top_n : int
        number of top images to plot
    class_index : int
        Index in classification array to utilize for scoring
    top_n_copy_loc : str
        Path to directory to copy the top n figures to

    """

    # Confirm a classification matrix is same size as list of images
    assert len(classification[:,class_index]) == len(imageFiles), \
        'Classification and imageFiles must be the same length'

    # Sort the images by classification score
    sorted_images_idx = np.argsort(classification[:,class_index])[::-1]

    images_to_plot = imageFiles[sorted_images_idx[:top_n]]

    plot_images(images_to_plot)

    # Copy files to top_n_copy_loc
    if top_n_copy_loc is not None:
        for file in images_to_plot:
            shutil.copy(file, top_n_copy_loc)
        print('Top images copied to {}'.format(top_n_copy_loc))

  # ncol = 4
  # nrow = int(np.ceil(top_n / ncol))

  # # Make sure column width is multiple of ncol
  # colWidth = ncol * np.ceil(top_n / ncol)
  # fig = plt.figure(figsize=(30,colWidth*2))
  # subplots = []

  # for i in range(top_n):
  #     # Copy image to directory
  #     if top_n_copy_loc is not None:
  #         shutil.copy(imageFiles[sorted_images_idx[i]], top_n_copy_loc)
  #     
  #     # Plot the images
  #     subplots.append(fig.add_subplot(nrow,ncol,i+1))
  #     subplots[-1].set_axis_off()
  #     subplots[-1].set_title(f"Rank = {i}; Score = {classification[sorted_images_idx[i],0]}",size=13)
  #     f = imageFiles[sorted_images_idx[i]]
  #     im = Image.open(f).convert('RGB')
  #     subplots[-1].imshow(im)

