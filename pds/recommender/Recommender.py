"""
This module contains tools to generate recommended data using a self-supervised learning model.

After constructing the object, the main method to use is 'recommend'

    Example Uage:
    
    recommender = Recommender(model_path, 'SIMCLR', pool_image_path, extra_features=extra_features, alpha_extra=1.0,
        metric='minkowski', p=2, n_jobs=-1)

    recommender.recommend(imageFiles, n_recommendations=10, extra_features=sample_extra_features, method='harmonic_knn')

    
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import shutil

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler
from scipy.stats import hmean
from scipy.spatial.distance import cdist
from scipy.special import erf

from ..self_supervised_learner.Evaluator import Evaluator
from ..utilities.image_utils import plot_images

class Recommender(Evaluator):
    """ Class used to return recommended data based on a SSL model and example data

    This inherents the self_supervised_learner.Evaluator.Evaluator class.

    """

    def __init__(self, 
            model_path, 
            technique, 
            pool_image_path, 
            extra_features=None,
            alpha_extra=1.0,
            metric='minkowski', 
            p=2, 
            kneighbors_expansion_factor=10, 
            gpu_index=0, 
            n_jobs=1, 
            limit_to_n_images=None):
        """ Constructs the Recommender object and loads the set of images to recommend from.

        There is an option to add in extra features. When they are passed in, they are concatenated to the end of the
        computed image embeddings (self.pool_embeddings). Then when calling the recommend method, you must pass in the same
        number of extra feartures.

        Parameters
        ----------
        model_path : 
            Path to the PyTorch Lightning model
        technique : 
            model technique used {SIMCLR, SIMSIAM or CLASSIFIER}
        pool_image_path  : str
            The path to the pool images to recommend from
        extra_features : np.array(nImages, nFeatures)
            If extra features are to be added to the SSL embedding then add them here as a matrix
        alpha_extra : float
            Scaling factor for relative length of extra features to embeddings.
            A value of 1.0 means equal scaling. 
            Greater than 1.0 means the extra features are set more distantly (hence stronger dependence).
        metric : str
            Distance metric to use
            see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.distance_metrics.html
            for options
        p : int
            Minkowski metric parameter
            For p = 1 => manhattan_distance (l1), for p = 2 => euclidean_distance (l2)
        kneighbors_expansion_factor : int
            When performing an harmonic mean nearest neighbors search, use this expansion factor when performing the initial search
            over each individual sample.
        gpu_index : int
            Specify the GPU index to use for processing
            Only useful for a multi-GPU machine
        n_jobs : int
            Number of CPU cores to use when recommending. 
            None means 1 core. -1 means using all cores.
        limit_to_n_images : int
            If not None then limit the number of images being recommended to N random samples from pool_image_path.
            This is mainly to speed up the algorithm for debugging purposes.

        """

        # Call the Evaluator class constructor
        super(Recommender, self).__init__(model_path, technique, gpu_index)

        self.pool_image_path = pool_image_path
        self.metric = metric
        self.p = p
        self.kneighbors_expansion_factor = kneighbors_expansion_factor

        # Evaluate the model on the passed images
        if limit_to_n_images is not None:
            self.pool_embeddings, pool_image_files, self.selected_images = self.evaluate_model(pool_image_path, limit_to_n_images=limit_to_n_images)
        else:
            self.pool_embeddings, pool_image_files = self.evaluate_model(pool_image_path, limit_to_n_images=limit_to_n_images)
            self.selected_images = np.arange(len(pool_image_files))

        # Add extra features
        if extra_features is not None:
            self.transformer = CustomRobustScaler(alpha_extra=alpha_extra)
            self.n_extra_features = np.shape(extra_features)[1]

            self.transformer.fit_to_embeddings(self.pool_embeddings)
            self.transformer.fit_to_extra(extra_features)

            extra_features_tr = self.transformer.transform_extra(extra_features)

            self.pool_embeddings = np.concatenate((self.pool_embeddings,extra_features_tr[self.selected_images,:]),1)
        else:
            self.transformer = None

        # Convert pool_image_files list into a ndarray
        self.pool_image_files = np.array(pool_image_files)
        self.n_images = len(self.pool_image_files)

        # Initialize a nearest neighbors object
        self.neighbors = NearestNeighbors(n_neighbors=10, metric=metric, p=p, n_jobs=n_jobs)
        self.neighbors.fit(self.pool_embeddings)

        return

    def recommend(self, 
            samples, 
            n_recommendations, 
            extra_features=None,
            repulsive_samples=None,
            repulsive_extra_features=None,
            beta_scale=1.0,
            method='harmonic_knn', 
            do_not_recommend_samples=True, 
            plotting=True, 
            recommendations_copy_loc=None,
            verbosity=True,
            ):
        """
        Returns N recommendations in aggregate over all sample images based on <method>.

        Available methods:
        'harmonic_knn': Returns the top n_recomendations images that are nearest to the sample images. Distance is
        computed by first computing the distance to each smaple image then taking the harmonic mean of all the distance.
        It then ranks based on this mean and returns the top n_recommendations

        Optionally, repulsive samples can also be passed. If using repulsors then <beta_scale> is used to set the
        relaitive strength of the repulsors on the recommendations. A value of 1.0 means equali weighting. Set greater
        than 1.0 for the repulsors to have greater relative strength.

        Parameters
        ----------
        samples : str list
            List of paths to sample images to find recommendations for
        n_recommendations : int
            Total number of recommendations to return
        extra_features : np.array(nImages, nFeatures)
            If extra features are to be added to the SSL embedding then add them here as a matrix
        repulsive_samples : str list
            List of paths to sample images to attempt to NOT recommend similar images
        repulsive_extra_features : np.array(nImages, nFeatures)
            If extra features are to be added to the SSL embedding then add them here as a matrix
        beta_scale : float
            Scaling factor for the beta parameter.
        method : str
            Recommendation method
            {'harmonic_knn'}
        do_not_recommend_samples : bool
            If True then do not recommend back any of the samples passed if they are in the Recommender.pool_image_path images used to
            construct the Recommender object.
        plotting : Bool
            If True then plot the recommendations
        recommendations_copy_loc : str
            If not None, then copy recommended images to this location
        verbosity : bool
            If True then use tqdm to show progress
        
        Returns
        -------
        recommendations : int ndarray(n_recommendations)
            Indices of the n recommendations in self.pool_image_files
        recommendation_dist : float ndarray(n_recommendations)
            Average distance each recommendation is to the sample ensemble

        """

        assert method == 'harmonic_knn', "Only 'harmonic_knn' method currently supported"

        if n_recommendations > self.n_images:
            raise Exception('Can not find n_recommendations, please reduce the number of recommendations at or below the total number of pool images')

        # Find the nearest neighbors for each sample within a distance
        if method == 'harmonic_knn':

            # Begin by finding kneighbors_expansion_factor times n_recommendations for each sample target
            neighbor_distance_matrix, neighbor_idx_matrix, sample_embedding = self.find_kneighbors(samples,
                    int(self.kneighbors_expansion_factor * n_recommendations), extra_features, verbosity=verbosity)

            #***
            # Flatten the arrays of neighbors, elliminate duplicates and compute the distance to all samples, in
            # preparation to taking the harmonic mean.

            # Flatten index array
            neighbor_idx_with_dups = neighbor_idx_matrix.flatten()

            # Elliminate duplicates
            neighbor_idx, unique_idx = np.unique(neighbor_idx_with_dups, return_index=True)
            neighbor_embeddings = self.pool_embeddings[neighbor_idx,:]

            # Elliminate sample images from returned neighbors
            # Get basename for all self.pool_image_files
            if do_not_recommend_samples:
                pool_image_files = [os.path.basename(filename) for filename in self.pool_image_files]
                # Get indices in the self.pool_image_files corresponding to any sample images
                sample_indices = []
                for sample_file in samples:
                    sample_indices.extend([idx for idx, filename in enumerate(pool_image_files) if
                                filename==os.path.basename(sample_file)])
                neighbor_idx = np.setdiff1d(neighbor_idx, sample_indices)
                neighbor_embeddings = self.pool_embeddings[neighbor_idx,:]

            # Compute distance to all sample images
            Y = cdist(neighbor_embeddings, sample_embedding, metric=self.metric, p=self.p)

            # Compute harmonic mean distance to each neighbor candidate
            hmean_dist = hmean(Y, axis=1)

            # Compute distance to all repulsors
            if repulsive_samples is not None:
                repulsive_embedding = self._compute_embeddings(repulsive_samples, extra_features=repulsive_extra_features, verbosity=verbosity)

                # Compute distance to all repulsive images
                Y_repulse = cdist(neighbor_embeddings, repulsive_embedding, metric=self.metric, p=self.p)

                # Compute harmonic mean distance from repulsors to each neighbor candidate
                hmean_dist_repulse = hmean(Y_repulse, axis=1)

                # Scale repulsor distance by factor
                # We need a maximum repulsive distance
                beta = self._compute_beta(sample_embedding, repulsive_embedding, beta_scale=beta_scale)
                hmean_dist_repulse_sigmoid = self._sigmoid(beta, hmean_dist_repulse)

                # Subtract repulsive distance to attactor distance
                hmean_dist -= hmean_dist_repulse_sigmoid

            # Pick the top n_recommendations neighbors
            pick_idx = np.argsort(hmean_dist)[0:n_recommendations]
            recommendations = neighbor_idx[pick_idx]
            recommendation_dist = hmean_dist[pick_idx]

        else:
            raise Exception('Unknown recommendation method')

        if plotting:
            plot_images(self.pool_image_files[recommendations], title_text='Dist', values=recommendation_dist)

        # Copy files to recommendations_copy_loc 
        if recommendations_copy_loc  is not None:
            images_to_copy = self.pool_image_files[recommendations]
            for file in images_to_copy:
                shutil.copy(file, recommendations_copy_loc)
            print('Recommended images copied to {}'.format(recommendations_copy_loc))


        return recommendations, recommendation_dist

    def find_kneighbors(self, samples, n_neighbors, extra_features=None, verbosity=True):
        """ Find n_neighbors nearest to the given samples

        Parameters
        ----------
        samples : str list
            List of sample images to find neighbors to
        n_neighbors : int
            Number of neighbors to return
        extra_features : np.array(nImages, nFeatures)
            If extra features are to be added to the SSL embedding then add them here as a matrix
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

        sample_embedding = self._compute_embeddings(samples, extra_features=extra_features, verbosity=verbosity)

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

        raise Exception('This method has not been kept up to date with new functionality.')

        # Evaluate the model in the passed samples
        sample_embedding, sample_pool_image_files = self.evaluate_model(samples, verbosity=verbosity)

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

        raise Exception('This method has not been kept up to date with new functionality.')

        assert method == 'knn', "Only 'nearest_neighbor' method currently supported"

        if method == 'knn':
            neighbor_distance, recommendations = self.find_kneighbors(samples, n_recommendations, verbosity)
        else:
            raise Exception('Unknown recommendation method')

        if plotting:
            for idx in np.arange(len(samples)):
                plot_images(self.pool_image_files[recommendations[idx,:]], title_text='Dist', values=neighbor_distance[idx,:])

               #plt.show()
                input('Press the Any key')
                plt.close()


        return recommendations

    def _compute_embeddings(self, images, extra_features=None, verbosity=True):
        """ Helper function to compute the embeddings of a selection of images, including extra features

        Parameters
        ----------
        images : str list
            List of sample images to find neighbors to
        extra_features : np.array(nImages, nFeatures)
            If extra features are to be added to the SSL embedding then add them here as a matrix
        verbosity : bool
            If True then use tqdm to show progress

        Returns
        -------
        sample_embedding : ndarray(nSamples, nEmbeddings)
            Embeddings of the given model on the specified dataset.

        """

        if extra_features is not None:
            assert self.n_extra_features == np.shape(extra_features)[1], \
                    'Must use same number of extra_features as when constructing Recommender object'

        # Evaluate the model in the passed images
        sample_embedding, _ = self.evaluate_model(images, verbosity=verbosity)

        # Add extra features
        if extra_features is not None:
            extra_features_tr = self.transformer.transform_extra(extra_features)
            sample_embedding = np.concatenate((sample_embedding,extra_features_tr),1)

        return sample_embedding

    def _compute_beta(self, attractive_embedding, repulsive_embedding, beta_scale=1.0):
        """ Takes the distribution of attractive and repulsive points in the embedding and computes a distance metric to
        be used to determine when one is near a repulsive image.

        Parameters
        ----------
        attractive_embedding : ndarray(nSamples, nEmbeddings)
            Embeddings of the attractors
        repulsive_embedding : ndarray(nSamples, nEmbeddings)
            Embeddings of the repulsors
        beta_scale : float
            Scaling factor for the beta parameter.

        Returns
        -------
        beta : float
            Distance metric to determine when one is near a repulsive image

        """

        '''
        #*************
        # Try clustering attractore and repulsors
        # Use DBSCAN to cluster datums
        from sklearn.cluster import DBSCAN

        # We need a distance metric for DBSCAN. Set to fraction of overall std of all embedding dimensions 
        # (which should already be normalized)
        stdLengthSample = np.median(np.nanstd(attractive_embedding, axis=0))
        stdLengthRepulse = np.median(np.nanstd(repulsive_embedding, axis=0))
        std_length = np.mean((stdLengthSample, stdLengthRepulse))
        # Set eps to 10% of eps average distance
        eps_frac = 0.1


        sample_clustering = DBSCAN(eps=eps_frac*std_length, min_samples=2).fit(attractive_embedding)
        '''        



        #*******
        # Compute the average distance between all attractive and repulsive points
        from scipy.spatial.distance import pdist
        Y_attract = pdist(attractive_embedding, metric=self.metric, p=self.p)
        Y_repulse = pdist(repulsive_embedding, metric=self.metric, p=self.p)

        beta = beta_scale * np.nanmedian(np.concatenate((Y_attract, Y_repulse)))

        return beta

    def _sigmoid(self, beta, x):
        """ Returns a truncated scaled sigmoid of the form:

            erf(x/beta)*beta

        where erf is the standard Error Function, beta is a distance metric which dictates the assymptotic 
        value of the sigmoid.

        Only accepts x >= 0

        So,
        y(x) = { beta if x => inf
               { 0 if x = 0
               { ERROR if x < 0 (would return -beta)

        """

        assert beta > 0.0, 'beta must be > 0.0'

        assert np.all(np.array(x) >= 0.0), 'x must be >= 0.0'

      # return (2*beta / (1 + np.exp(-growth*x))) - beta

        return erf(x/beta)*beta

#*************************************************************************************************************

class CustomRobustScaler(RobustScaler):
    """ This is a custom scikit-learn RobustScalar class to allow one to fit normalization values to a matrix of
    features then apply the median of the normalziation values to extra features.

    """

    def __init__(self,
            *,
            with_centering=True,
            with_scaling=True,
            quantile_range=(25.0, 75.0),
            copy=True,
            unit_variance=False,
            alpha_extra=1.0,
            ):
        """

        Parameters
        ----------
        alpha_extra : scaling factor for relative length of extra features to embeddings.
            A value of 1.0 means equal scaling. > 1.0 means the extra features are set more distant.

        with_centering : bool, default=True
            If `True`, center the data before scaling.
            This will cause :meth:`transform` to raise an exception when attempted
            on sparse matrices, because centering them entails building a dense
            matrix which in common use cases is likely to be too large to fit in
            memory.
        
        with_scaling : bool, default=True
            If `True`, scale the data to interquartile range.
        
        quantile_range : tuple (q_min, q_max), 0.0 < q_min < q_max < 100.0, \
            default=(25.0, 75.0)
            Quantile range used to calculate `scale_`. By default this is equal to
            the IQR, i.e., `q_min` is the first quantile and `q_max` is the third
            quantile.
        
        copy : bool, default=True
            If `False`, try to avoid a copy and do inplace scaling instead.
            This is not guaranteed to always work inplace; e.g. if the data is
            not a NumPy array or scipy.sparse CSR matrix, a copy may still be
            returned.
        
        unit_variance : bool, default=False
            If `True`, scale data so that normally distributed features have a
            variance of 1. In general, if the difference between the x-values of
            `q_max` and `q_min` for a standard normal distribution is greater
            than 1, the dataset will be scaled down. If less than 1, the dataset
            will be scaled up.
        
        Attributes
        ----------
        center_ : array of floats
            The median value for each feature in the training set.
        
        scale_ : array of floats
            The (scaled) interquartile range for each feature in the training set.
        
               *scale_* attribute.
        
        n_features_in_ : int
            Number of features seen during :term:`fit`.
        
        feature_names_in_ : ndarray of shape (`n_features_in_`,)
            Names of features seen during :term:`fit`. Defined only when `X`
            has feature names that are all strings.
        

        """


        # Call the RobustScaler superclass init method
        super(CustomRobustScaler, self).__init__(
            with_centering=with_centering,
            with_scaling=with_scaling,
            quantile_range=quantile_range,
            copy=copy,
            unit_variance=unit_variance,
            )

        self.alpha_extra = alpha_extra

        self.medians_computed = False
        self.extra_scale_computed = False

        return
    
    def fit_to_embeddings(self, X):
        """ Fits the RobustScaler values to the given embeddings matrix.

        It then stores the median center and scale values for use to apply to the extra features

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            The data used to compute the median and quantiles
            used for later scaling along the features axis.

        """

        # First compute the center and scale using the standard RobustScaler method
        self.fit(X)

        # Compute the median values
        self.median_center_ = np.nanmedian(self.center_)
        self.median_scale_ = np.nanmedian(self.scale_)

        self.medians_computed = True

    def fit_to_extra(self, X_extra):
        """ Fits the tranformer values to the extra features. 

        Also scales the transformer values to be of the same scale as the embeddings determined by fit_to_embeddings.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            The data used to compute the median and quantiles
            used for later scaling along the features axis.

        """

        self.n_extra_features = np.shape(X_extra)[1]

        # Create a temprary RobustScaler object to compute the values then store in this object
        temp_transformer = RobustScaler()

        temp_transformer.fit(X_extra)
        self.extra_center_ = temp_transformer.center_
        self.extra_scale_ = temp_transformer.scale_

        self.extra_scale_computed = True


    def transform_extra(self, X_extra):
        """ Applies the median tranform values to the extra features X.

        Parameters
        ----------
        X_extra : {array-like} of shape (n_samples, n_features)
            The data used to scale along the specified axis.

        Returns
        -------
        X_tr : {ndarray} of shape (n_samples, n_features)
            Transformed array.

        """

        assert self.medians_computed, 'Must call fit_to_embeddings for this method'
        assert self.extra_scale_computed, 'Must call fit_to_extra for this method'

        X_tr = copy.copy(X_extra)

        # First transform to each extra features robust scale
        X_tr -= self.extra_center_
        
        X_tr /= self.extra_scale_

        
        # Then transform to the median embeddings scale
        # Do this before displacing the center
        X_tr *=  np.full((1, self.n_extra_features), self.median_scale_)

        # Scale by alpha
        # Do this before displacing the center
        X_tr *= np.full((1, self.n_extra_features), self.alpha_extra)

        X_tr += np.full((1, self.n_extra_features), self.median_center_)
        

        return X_tr

