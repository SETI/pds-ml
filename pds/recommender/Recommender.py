# This module contains tools to egenrate recommended data using a self-supervised learning model

import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy.stats import hmean
from scipy.spatial.distance import cdist
from tqdm import tqdm

from ..self_supervised_learner.Evaluator import Evaluator
from ..utilities.image_utils import plot_images

class Recommender(Evaluator):
    """ Class used to return recommended data based on a SSL model and example data

    This inherents the self_supervised_learner.Evaluator.Evaluator class.

    """

    def __init__(self, 
            model_path, 
            technique, 
            image_path, 
            metric='minkowski', 
            p=2, 
            kneighbors_expansion_factor=10, 
            gpu_index=0, 
            n_jobs=1, 
            limit_to_n_images=None):
        """ Constructs the Recommender object and loads the set of images to recommend from..

        Parameters
        ----------
        model_path : 
            Path to the PyTorch Lightning model
        technique : 
            model technique used {SIMCLR, SIMSIAM or CLASSIFIER}
        image_path  : str
            The path to the pool images to recommend from
        metric : str
            Distance metric to use
            see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.distance_metrics.html
            for options
        p : int
            Minkowski metric parameter
            For p = 1 => manhattan_distance (l1), for p = 2 => euclidean_distance (l2)
        kneighbors_expansion_factor : int
            When performing a harminic mean nearest neighbors search, use this expansion factor when performing the initial search
            over each individual sample.
        gpu_index : int
            Specify the GPU index to use for processing
            Only useful for a multi-GPU machine
        n_jobs : int
            Number of CPU cores to use when recommending. 
            None means 1 core. -1 means using all cores.
        limit_to_n_images : int
            If not None then limit the number of images being recommended to N

        """

        # Call the Evaluator class constructor
        super(Recommender, self).__init__(model_path, technique, gpu_index)

        self.image_path = image_path
        self.metric = metric
        self.p = p
        self.kneighbors_expansion_factor = kneighbors_expansion_factor

        # Evaluate the model on the passed images
        self.embedding, image_files = self.evaluate_model(image_path, limit_to_n_images=limit_to_n_images)
        # Convert image_files list into a ndarray
        self.image_files = np.array(image_files)
        self.n_images = len(self.image_files)

        # Initialize a nearest neighbors object
        self.neighbors = NearestNeighbors(n_neighbors=10, metric=metric, p=p, n_jobs=n_jobs)
        self.neighbors.fit(self.embedding)

        return

    def find_kneighbors(self, samples, n_neighbors, verbosity=True):
        """ Find n_neighbors nearest to the given samples

        Parameters
        ----------
        samples : str list
            List of sample images to find neighbors to
        n_neighbors : int
            Number of neighbors to return
        verbosity : bool
            If True then use tqdm to show progress

        Returns
        -------
        neighbor_distance : float ndarray(n_samples, n_neighbors)
            Distance each neighbor is to each sample
        neighbor_idx : int ndarray(n_samples, n_neighbors)
            Indices of the n nearest neighbors for each sample
        sample_embedding : ndarray(nSamples, nEmbeddings)
            Embeddings of the given model on the specified dataset.

        """

        if n_neighbors > self.n_images:
            n_neighbors = self.n_images

        # Evaluate the model in the passed samples
        sample_embedding, sample_image_files = self.evaluate_model(samples, verbosity=verbosity)

        # Find nearest neighbors for each sample
        neighbor_distance, neighbor_idx = self.neighbors.kneighbors(sample_embedding, n_neighbors=n_neighbors)

        return neighbor_distance, neighbor_idx, sample_embedding

    def find_radius_neighbors(self, samples, radius, verbosity=True):
        """
        Returns all neighbors within radius for each of the given samples

        Parameters
        ----------
        samples : str list
            List of sample images to find neighbors to
        radius : float
            Euclidean Radius within to find all neighbors
        verbosity : bool
            If True then use tqdm to show progress

        Returns
        -------
        neighbor_distance : ndarray of shape (n_samples,) of arrays of floats
            Array representing the distances to each point.
        nearest_neighbors : ndarray of shape (n_samples,) of arrays of ints
            An array of arrays of indices of the approximate nearest points from the population matrix that lie within a
            ball of size radius around the query sample points.
    
        """

        # Evaluate the model in the passed samples
        sample_embedding, sample_image_files = self.evaluate_model(samples, verbosity=verbosity)

        # Find nearest neighbors for each sample within radius
        neighbor_distance, nearest_neighbors = self.neighbors.radius_neighbors(sample_embedding, radius=radius)

        return neighbor_distance, nearest_neighbors


    def recommendations_per_sample(self, samples, n_recommendations, method='knn', plotting=True, verbosity=True):
        """
        Returns N recommendations per sample image based on <method>.

        available methods:
        'knn' : k nearest neighbors, returns the n_recommendations nearest to each of the sanmple images. Note that this
        can result in the same images returned for each sample image.

        Parameters
        ----------
        samples : str list
            List of paths to sample images to find recommendations for
        n_recommendations : int
            Number of recommendations to return
        method : str
            Recommendation method
            {'knn'}
        plotting : Bool
            If True then plot the recommendations
        verbosity : bool
            If True then use tqdm to show progress
        
        Returns
        -------
        recommendations : int ndarray(n_samples, n_recommendations)
            Indices of the n recommendations for each sample

        """

        assert method == 'knn', "Only 'nearest_neighbor' method currently supported"

        if method == 'knn':
            neighbor_distance, recommendations = self.find_kneighbors(samples, n_recommendations, verbosity)
        else:
            raise Exception('Unknown recommendation method')

        if plotting:
            for idx in np.arange(len(samples)):
                plot_images(self.image_files[recommendations[idx,:]], title_text='Dist', values=neighbor_distance[idx,:])

               #plt.show()
                input('Press the Any key')
                plt.close()


        return recommendations

    def recommendations(self, samples, n_recommendations, method='harmonic_knn', do_not_recommend_samples=True, plotting=True, verbosity=True):
        """
        Returns N recommendations in aggregate over all sample images based on <method>.

        Available methods:
        'harmonic_knn': Returns the top n_recomendations images that are nearest to the sample images. Distance is
        computed by first computing the distance to each smaple image then taking the harmonic mean of all the distance.
        It then ranks based on this mean and returns the top n_recommendations


        Parameters
        ----------
        samples : str list
            List of paths to sample images to find recommendations for
        n_recommendations : int
            Total number of recommendations to return
        method : str
            Recommendation method
            {'harmonic_knn'}
        do_not_recommend_samples : bool
            If True then do not recommend back any of the samples passed if they are in the Recommender.image_path images used to
            construct the Recommender object.
        plotting : Bool
            If True then plot the recommendations
        verbosity : bool
            If True then use tqdm to show progress
        
        Returns
        -------
        recommendations : int ndarray(n_recommendations)
            Indices of the n recommendations in self.image_files
        recommendation_dist : float ndarray(n_recommendations)
         Harmonic distance each recommendation is to the sample ensemble

        """

        assert method == 'harmonic_knn', "Only 'harmonic_knn' method currently supported"

        if n_recommendations > self.n_images:
            raise Exception('Can not find n_recommendations, please reduce the number of recommendations at or below the total number of pool images')

        # Find the nearest neighbors for each sample within a distance
        if method == 'harmonic_knn':

            # Begin by finding kneighbors_expansion_factor times n_recommendations for each sample target
            neighbor_distance_matrix, neighbor_idx_matrix, sample_embedding = self.find_kneighbors(samples,
                    int(self.kneighbors_expansion_factor * n_recommendations), verbosity=verbosity)

            #***
            # Flatten the arrays of neighbors, elliminate duplicates and compute the distance to all samples, in
            # preparation to taking the harmonic mean.

            # Flatten index array
            neighbor_idx_with_dups = neighbor_idx_matrix.flatten()

            # Elliminate duplicates
            neighbor_idx, unique_idx = np.unique(neighbor_idx_with_dups, return_index=True)
            neighbor_embeddings = self.embedding[neighbor_idx,:]

            # Elliminate sample images from returned neighbors
            # Get basename for all self.image_files
            if do_not_recommend_samples:
                image_files = [os.path.basename(filename) for filename in self.image_files]
                # Get indices in the self.image_files corresponding to any sample images
                sample_indices = []
                for sample_file in samples:
                    sample_indices.extend([idx for idx, filename in enumerate(image_files) if
                                filename==os.path.basename(sample_file)])
                neighbor_idx = np.setdiff1d(neighbor_idx, sample_indices)
                neighbor_embeddings = self.embedding[neighbor_idx,:]

            # Compute distance to all sample images
            Y = cdist(neighbor_embeddings, sample_embedding, metric=self.metric, p=self.p)

            # Compute harmonic mean distance to each neighbor candidate
            hmean_dist = hmean(Y, axis=1)

            # Pick the top n_recommendations neighbors
            pick_idx = np.argsort(hmean_dist)[0:n_recommendations]
            recommendations = neighbor_idx[pick_idx]
            recommendation_dist = hmean_dist[pick_idx]

        else:
            raise Exception('Unknown recommendation method')

        if plotting:
            plot_images(self.image_files[recommendations], title_text='Dist', values=recommendation_dist)

        return recommendations, recommendation_dist


                
