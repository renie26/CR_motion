This folder contatins scripts for similarity modelling leveraging both supervised calssification and unsupervised clustering.


# Supervised Similarity Calculation
## Methodology
In order to compute the similarity between videos, each video was considered as a multidimensional time series of different frames.
We included three types of similarity calculation: Dynamic Time Warping (DTW), soft-min Dynamic Time Warping and Radial Basis Function kernel. We also included a further method which is based on averaging the results of the other methods.

In addition, as an extra part for further discussion and research, we include a similarity search based on clustering. Using k-means clustering with DTW distance, we assign to an input video a cluster of videos to which it belongs. However, the size of this cluster will not necessarily be $k$, so this is not a solution to the original problem.



## ```similarity_calculation.py```

This file contains different functions to compute the similarity between two videos. In particular, 

- ```similarity_dtw``` computes the similarity using the dynamic time warping (DTW) algorithm. 
- ```similarity_soft_dtw``` uses the soft-min approach to DTW as described by Cuturi and Blondel (2017)
- ```similarity_rbf_based``` makes use of the radial basis function kernel

Then, we define the function ```retrieve_top_k```, which returns the $k$ videos that are most similar to the input video, given the input dataset. This function allows to select any of the previously mentioned types of similarity function. The function calculates the similarity of the input video to all the other videos in the dataset, and returns the $k$ most similar ones.

We also included a feedback function, ```Feedback```, which returns the F1 score over the $k$ videos retrieved. This is calculated using the annotations provided together with each video.

Finally, the function ```retrieve_top_k_clusters``` returns the cluster to which the video belongs. This cluster might not have size $k$. The number of clusters is calculated as $\frac{\text{size of dataset}}{k}$, and therefore the average size of the cluster is $k$. This functions uses the KMeans algorithm from ```tslearn``` to find the clusters.

# Demo on ```posecuts_sideviews```

The file ```Demo_run.py```executes the similarity calculation based both on dynamic time warping and  a Radial basis function.
It will output a graph that displays the performance of the model on ```posecuts_sideviews```.

Similarly, ```Demo_run_clustering.py``` enacts the similarity model based on clustering, and shows its performance with a graph.



# Unsupervised Similarity Clustering
