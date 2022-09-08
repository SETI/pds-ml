# This module contains tools to egenrate recommended data using a self-supervised learning model

import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import matplotlib.pyplot as plt

from ..self_supervised_learner.Evaluator import Evaluator
from ..utilities.image_utils import plot_images

class Recommender(Evaluator):
    """ Class used to return recommended data based on a SSL model and example data

    It inhirents the self_supervised_learner.Evaluator.Evaluator class.

    """

    def __init__(self, model_path, technique, image_path, gpu_index=0):
        """ Constructs the Recommender object where the recommend method is specified.

        Parameters
        ----------
        model_path : 
            Path to the PyTorch Lightning model
        technique : 
            model technique used {SIMCLR, SIMSIAM or CLASSIFIER}
        image_path  : str
            The path to the images to recommend from
        gpu_index : int
            Specify the GPU index to use for processing
            Only useful for a multi-GPU machine

        """

        # Call the Evaluator class constructor
        super(Recommender, self).__init__(model_path, technique, gpu_index)

        self.image_path = image_path

        # Evaluate the model on the passed images
        self.embedding, image_files = self.evaluate_model(image_path)
        # Convert image_files list into a ndarray
        self.image_files = np.array(image_files)

        # Initialize a nearest neighbors object
        self.neighbors = NearestNeighbors(n_neighbors=10)
        self.neighbors.fit(self.embedding)

        return

    def find_kneighbors(self, samples, n_neighbors):
        """ Find n_neighbors nearest to the given samples

        Parameters
        ----------
        samples : str list
            List of sample images to find neighbors to
        n_neighbors : int
            Number of neighbors to return

        Returns
        -------
        nearest_neighbors : int ndarray(n_samples, n_neighbors)
            Indices of the n nearest neighbors for each sample

        """

        # Evaluate the model in the passed samples
        sample_embedding, sample_image_files = self.evaluate_model(samples)


        # Find nearest neighbors for each sample
        neighbor_distance, nearest_neighbors = self.neighbors.kneighbors(sample_embedding, n_neighbors=n_neighbors)

        return neighbor_distance, nearest_neighbors

    def recommendations(self, samples, n_recommendations, method='nearest_neighbors', plotting=True):
        """
        returns N recommendations based on <method> of the sample images.

        Parameters
        ----------
        samples : str list
            List of sample images to find recommendations for
        n_recommendations : int
            Number of recommendations to return
        method : str
            Recommendation method
            {'nearest_neighbors'}
        plotting : Bool
            If True then plot the recommendations
        
        Returns
        -------
        recommendations : int ndarray(n_samples, n_recommendations)
            Indices of the n recommendations for each sample

        """

        assert method == 'nearest_neighbors', "Only 'nearest_neighbor' method currently supported"

        if method == 'nearest_neighbors':
            neighbor_distance, recommendations = self.find_kneighbors(samples, n_recommendations)

        else:
            raise Exception('Unknown recommendation method')

        if plotting:
            for idx in np.arange(len(samples)):
                plot_images(self.image_files[recommendations[idx,:]], title_text='Dist', values=neighbor_distance[idx,:])

               #plt.show()
                input('Press the Any key')
                plt.close()


        return
                
