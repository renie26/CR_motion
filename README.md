# Motion Similarity Based Video Search

The code, implemented in Python, is produced from the EPFL-EDDH Semester Project #1 of Y.H., where a video search framework was proposed to retrieve the top-k (supervised) or nearest (unsupervised) videos to the query based on motion similarity. Specifically, this workflow treats features as motion time series and applies well to vary-in-lenth/-speed video dataset.


## Table of contents

<!--ts-->
   * [Table of contents](#table-of-contents)
   * [Introduction](#introduction)
   * [Acknowledgement](#acknowledgement)
   * [Prerequisites](#prerequisites)
      * [Libraries](#libs)
      * [Data](#data)
   * [Methodology](#prerequisites)
      * [Pre-processing](#preprocess)
      * [Feature Construction](#feature)
      * [Similarity Modelling](#similarity)
   * [Contact & Contribute](#contact)
   * [License](#license)
<!--te-->



## Introduction
The proposed framework works as follows:
	First, we traverse all the videos in the dataset to extract the per-x-frame features of each as sequences of body keypoints. 
	Then, we expand the keypoint sequences to a higher-dimensional feature space and treat them as data time series. 
	Afterwards, we 
		a) (supervised) train the similarity model using respective method e.g. DTW, Soft-DTW and RBF, and then average the computed models towards an optimal status. 
		b) (unsupervised) cluster the videos based on series distance (e.g. DTW clustering) while providing the flexibility of only computing a subset of parameters by selecting one of the predefined models, e.g. limb-only. 
Given a new query video, our workflow is capable to process its motion features, compute the similarity, and return the top-k or nearest results effectively and rapidly. 


## Acknowledgement

I'd like to recognize the contribution of my collaborating students:
* [**F. M. Seydou**](https://ch.linkedin.com/in/fadel-mamar-seydou-460a43197s), A. Fornaroli and R. Mocan contributed their implementation of supervised methodologies, including DTW-, RBF- and model averaging approaches. 
* [**D. Cian**](https://gitlab.com/davidcian) J. Quiroz and A. Aboueloul contributed various methods to inspect the quality of classification and clustering.


## Prerequisites

### Libraries

Depending on your preferable solution to keypoint extraction:
- [**OpenPose**](https://github.com/CMU-Perceptual-Computing-Lab/openpose) or [**PoseNet**](https://github.com/tensorflow/tfjs-models/tree/master/posenet) and their respective dependencies

If you choose to implement supervised classification:
- [**scikit-learn**](https://scikit-learn.org/stable/)
- [**tslearn**](https://tslearn.readthedocs.io/en/stable/)
If you choose to implement unsupervised clustering:
- [**dtaidistance**](https://dtaidistance.readthedocs.io/en/latest/usage/dtw.html)

Other common dependencies include:
- ```scipy```
- ```pandas```
- ```natsort```
- ```numpy```
- ```pickle```

....

In case of missing package, simply use [**pip**](https://pip.pypa.io/en/stable/reference/pip_install/) to install them! 


### Data

- A folder of videos of any format that is processible by the chosen keypoint extraction approach.
- If you decide to proceed with classification, you will need to form an annotation file with the tag describing 'motion class' for each video in the training set and validation set. 

##### Data Example

Due to confidentiality, we only provide samples of data from different stages:
- ```../data/frontview/json_per_video_sample```: sample data extracted using OpenPose in the form of JSON files, which will be utilized by the clustering method.
- ```../data/frontview```: ".txt" + ".csv" are ready to feed the classification method, where the features are in the format of OpenPose extraction.
- ```../data/sideview```: ".txt" + ".csv" are ready to feed the classification method, where the features are in the format of PoseNet extraction.


## Methodology

### Pre-processing

#### Feature Construction  

```../scripts/feature_processing``` contains all the essential functions to construct motion features out of the extracted raw data. Specifically, function ```dataset_preprocessing``` performs the following operations:
- Standardization (using ```sklearn```)
- Feature expansion, which includes cosine, sine, square root and polynomial expansion of degree 3.
- Angular velocity and acceleration calculation, which are important to analyse the motion that occurs in a video. Velocities are calculated as the angle difference between frames and accelerations are found as difference in velocities.
- PCA Dimensionality reduction (using ```sklearn```)


#### Additional Work

You can also find assisting scripts in this folder ```../scripts/pre-prosessing```, which perform various pre-processing functions including original keypoint extraction using OpenPose and PoseNet, trajectory and joint angle calculation, data transformation, time-series calculation and harmonization for different similarity modelling approaches.


## Similarity Modelling

The folder ```../clustering/``` and ```../classification/``` contain scripts for similarity modelling using respective approaches.

### Classification

```../classification/similarity_calculation.py``` contains the core functions to compute similarity between videos, where each video is considered as a multidimensional time series of different frames.

- ```similarity_dtw``` computes the similarity using Dynamic Time Warping (DTW). 
- ```similarity_soft_dtw``` uses the soft-min approach to DTW (Cuturi and Blondel, 2017)
- ```similarity_rbf_based``` adopts the usage of radial basis function kernel

The retrieval function ```retrieve_top_k``` will return the $k$ videos that are most similar to the query given input dataset. This function allows to select any of the aforementioned similarity functions or switch to model averaging, and calculates accordingly.

The feedback function ```Feedback``` will return the F1 score over the $k$ videos retrieved. The F1 metric is formed as the correctness of ```motion tags``` provided by the annotation file.

- ```Demo_run.py```executes the similarity computation by averaging DTW and RBF methods and displays a graph of performance on ```../data/sideview```.


#### Clustering

```../clustering/..._dtwcluster.py``` contains the core functions to cluster the videos based on the similarity distance (using ```dtaidistance```) between motion times series.

```../clustering/..._search.py``` executes the workflow of processing the query video (called through argv[1]), computing similarity with the selected subset of parameters (called by argv[2]) and returning the members within respective cluster. 

- ```../clustering/byJointAngles....py``` models the motion features using body joint angles. 
- ```../clustering/byTrajectories....py``` models the motion features using keypoint trajectories. 


## Contact & Contribute

If you want to discuss more on the project or exchange your ideas, please reach me at <rainie dot hym at gmail dot com>. 

If you want to contribute to this project, please follow the following steps:

1. Fork this repository.
2. Create a branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`
4. Push to the original branch: `git push origin <project_name>/<location>`
5. Create the pull request.

Alternatively see the GitHub documentation on [creating a pull request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).


## License

This project uses the [MIT License](<https://github.com/renie26/CR_motion/blob/main/LICENSE>).
