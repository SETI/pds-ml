# pds-ml
This is a repository created to develop machine learning tools for NASA's Planetary Data System.

The repo contains a custom recommender system tool that utilizes a self-supervised learning methodology
to aid users in identifying data of interest across image data sets.

Self-Supervised Learning is a technique where instead of utilizing a human labeled training data set to perform
supervised learning, the SSL technique probes the data on its own, generating its own self-described representation of
the data. A main advantage being that the characteristics of the data of interest does not need to be defined a priori.
It is also possible for the SSL to identify characteristics of the data not expected by humans. Our SSL algorithm is
based on a method developed by Google Research called SimCLR [SimCLR](https://github.com/google-research/simclr). This
method was adapted by the Frontier Development Laboratory’s SpaceML team, who developed a tool to take a selection of
images and apply the SimCLR method to create an SSL representation
[](https://github.com/spaceml-org/Self-Supervised-Learner).  We adapted and expanded upon the FDL tool for our purposes.
Our moddfied SSL tool is contained in this package.

Our developed tool takes the trained SSL model as input plus a pool of images that it evaluates using the model. The
evaluated model output is stored as a hyper-dimensional representation of all the images. A user then provides a small
number of sample images. Then using a custom semi-supervised learning method, images are recommended that are similar to
those provided by the user. A user provides sample images in two categories: attractors and repulsers. The recommender
then attempts to recommend images similar to the attractors but distant to the repulsers.  This technique allows for an
iterative approach, where the user begins with a small number of attractors (as few as a single image). The recommender
then returns a user-specified number of recommendations. The user can then either accept the recommendations or label
the returned images as either good or bad. The labeled images are then passed back to the recommender, are combined with
the images from the previous iteration and then the recommender returns a better set of recommended images. This
iterative process can continue until the user accepts the recommendations.

We implemented a semi-supervised learning method which utilizes a custom harmonic mean k-nearest neighbors approach.
The method will be described in a future paper.


# Installation

This package is in active development with
no formal releases. There are also nor formal requirements list (This is a TODO!).
It is recommended to install in development mode. There are different ways to do this. A simple way is to use the `-e`
option for pip. After activating your Python environment, in the top level directory of this repo type
```
pip install -e .
```
You can easily create a conda environment that runs this package with the conda yml file located here:
```
pds-ml/system/env/environment_A5000.yml
```

# Usage

This section is yet to be completed!
