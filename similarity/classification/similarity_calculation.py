"""
Acknowledgement: 
Implementation by Fadel Mamar Seydou, Alessandro Fornaroli, Razvan-Florin Mocan
Motion-based Similarity Search in Videos of Confucian Rituals" (EPFL, 2020)
"""  
import numpy as np
import math
import time
import tslearn.metrics   #if tslearn is not installed,--> $ pip install tslearn
import pandas as pd
from scipy.spatial.distance import cdist
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans, KernelKMeans

#---------------------------------Similarity calculation---------------------------------------------------------------
def similarity_dtw(video1,video2,metric="cosine",itakura=False,itakura_max_slope=2.,sakoe_chiba=True,sakoe_chiba_radius=10):
    """   This method will compute the similarity between the two videos
    
    video1:  the pre-processed features of video1 is a numpy array of shape (size1*d)
                                                        where each column is a frame
    
    video2: the pre-processed features of video2 with is a numpy array of shape (size2*d)
    
    metric : specifies the metric to use for calculating the similarity using DTW
    if metric is not given then it assumes an euclidean distance
        
        For good performance, we advise to use one of the these: ‘chebyshev’, ‘cityblock’,
        ‘correlation’, ‘cosine’, ‘seuclidean’, ‘sqeuclidean’.
        
        it follows  'scipy.spacial.distance.pdist' as it is the underlying function being run.
    
    itakura (boolean): specifies if the itakura algorithm is to be used
    
    sakoe_chiba (boolean): specifies if the itakura algorithm is to be used
    sakoe_chiba_radius (int): the restriction window size along the diagonal
    
        
    """
    
    #----Computing the cost 
    
    #--Restricting the path to resemble a parallelogram --> itakura method
    if itakura :
        _,cost = tslearn.metrics.dtw_path_from_metric(video1,video2,metric,global_constraint="itakura",
                                   itakura_max_slope=itakura_max_slope)
    
    #--Restricting the path to resemble a band --> sakoe_chiba method
    if sakoe_chiba :
        _,cost = tslearn.metrics.dtw_path_from_metric(video1,video2,metric,global_constraint="sakoe_chiba",
                                   sakoe_chiba_radius=sakoe_chiba_radius)
        
    #--Using the traditional formulation that goes through the whole distance matrix
    else :
        _,cost = tslearn.metrics.dtw_path_from_metric(video1, video2)
    
    return cost


def similarity_soft_dtw (video1, video2,gamma=1):
    """   
    This method uses a novel algorithm that uses a soft-min approach to dynamic time wrapping 
    
    video1:  the pre-processed features of video1 it's a numpy array of size (n*m)
                                                        where each column is a frame
    
    video2: the pre-processed features of video2 with similar shape as video1
    
    when gamma= 0 it corresponds to the standard DTW
    
    """
            
    score = tslearn.metrics.soft_dtw(video1, video2,gamma)
    
    
    return score

def similarity_rbf_based(video1,video2,metric='sqeuclidean'):
    """   This method uses a similar approach to the kernelized RBF function
           It computes a pairwise distance using scipy.spatial.distance.cdist 
           
           Video1 : features as numpy array
           
           Video2: features as numpy array
           
           metric: the distance modelling
               For good performances, we advise :  ‘chebyshev’, ‘cosine’,‘euclidean’,
               ‘seuclidean’,‘sqeuclidean’,'cityblock', ‘correlation’.
           Output: 
               the similarity score: if it's close to 1, then there is high similarity '
    """
     
    sigma = 1
    gamma = 0.5/sigma**2
    distance =  cdist(video1,video2,metric=metric)
    distance = np.mean(distance)
    return  np.exp(-gamma*distance)

#---------------------------------Retrieving top k--------------------------------------------------------------------------

def retrieve_top_k(input_video ,input_video_name, lookUp_dataset, k , csv_annotation_file_path,model_averaging=True ):
    """  
    Inputs:
        video1 : it's the input (numpy.array) for which we are looking for k similar videos
                It should be preprocessed
        
        lookUp_dataset: a dictionnary where:
                the key --> is the name of the video in the folder 
                the value --> is the pre-processed features in the video
    
        k : (int) is the number of video to retrieve
        
        similarity_function: use the function determined above similarity_dtw or similarity_soft_dtw
            
        model_averaging:  (bool) specify whether to output an upper bound on the performance or not
    
    Outputs:
        top_k_videos : names of most similar videos (scalar)
        similarity_score : mean F1 score (scalar)
            
            NB: the method uses all three methods to determine the most similar videos.
            When model_averaging is false, the best results among the three methods is chosen.
            When model_averaging is true, the results from the methods are merged and then sorted
                        using the feedback_method, which gives an upper bound on the performance.
            
        
    """
    
    assert type(k) is int, "k must be an integer"
    assert k <= len(lookUp_dataset.values())," k is too large, use a smaller value for k."
    
    #--Varibales and functions
    similarity_functions = [similarity_rbf_based, similarity_soft_dtw, similarity_dtw]
    output_rbf = dict()
    output_dtw = dict()
    output_softDTW = dict()
    
    #--iterating over values in lookUp_dataset
    for index,func in enumerate(similarity_functions) :
        scores = []
        video_names = []
        for video_name, video in lookUp_dataset.items():
            scores.append(func(input_video, video))
            video_names.append(video_name)
        
        #--sorting and taking the K small scores
        sorted_k = [[name,score] for score,name in sorted(zip(scores,video_names))][:k]
        
        if (func == similarity_rbf_based):
            sorted_k = sorted_k[::-1]   #the rbf-based method gives a score betwen 0 and 1, so we reverse the array
            
        keys = [sorted_k[i][0] for i in range(k)]      #--names
        values = [sorted_k[i][1] for i in range(k)]    #--scores
        
        #--Storing results in a dictionary
        dummy = {keys[i]: values[i] for i in range(k)}
        
        #--Outputting the results
        if(index == 1):
            output_rbf.update(dummy)
        if(index == 0):
            output_dtw.update(dummy)
        if (index == 2):
            output_softDTW.update(dummy)
            
    #----Taking the best from the three methods ---     
    if(model_averaging == True):
        all_selected_names = list.copy(list(output_rbf.keys()))
        for i in list(output_dtw.keys()):
            all_selected_names.append(i)
        for i in list(output_softDTW.keys()):
            all_selected_names.append(i)
             
        score_best,names_best = Feedback(csv_annotation_file_path,
                                         input_video_name,all_selected_names, k = k  )
          
        return score_best,names_best 
    
    #--In case the model averaging is False
    else:
        score_rbf,names_rbf  = Feedback(csv_annotation_file_path,
                                         input_video_name,list(output_rbf.keys()), k = k  )
        
        score_dtw, names_dtw = Feedback(csv_annotation_file_path,
                                         input_video_name,list(output_dtw.keys()), k = k  )
        
        score_softdtw, names_softdtw = Feedback(csv_annotation_file_path,
                                         input_video_name,list(output_softDTW.keys()), k = k  )
        
        score_selected, names_selected = sorted(zip([score_rbf,score_dtw,score_softdtw],[names_rbf,names_dtw,names_softdtw]))[-1]
        
        return score_selected, names_selected
    
#------------------------------------ Feedback function----------------------------------------------
def f_1_score(list1,list2):
    """   Compare the motion similarity from annotations
    Inputs:
        list1 : numpy.array containing strings 
        list2 : numpy.arraycontaining  strings
    Output:
        f_1_score in %
    """
    
    intersection = np.intersect1d(list1,list2)
    precision = intersection.shape[0]/(list1.shape[0])
    recall =  intersection.shape[0]/(list2.shape[0])
    
    if(recall+precision != 0):
        fscore = 2*(precision*recall)/(recall+precision)
    else:
        fscore = 0
        
    return fscore*100

def extract_motion_annotations(csv_file_path):
    """  
            Input: 
                csv_file_path (string): the path to the csv file
            
            Output: 
            (dict) a dictionnary that looks like
            {names: motions}  with names as keys and motions as values
                              motions are  numpy.array with the motions and the view 'front' or 'side'  
    
    """
    #--Retrieving the names
    names = np.genfromtxt(csv_file_path,delimiter=',',skip_header=1,
                          usecols=0,encoding='utf8',dtype='str').tolist() 
    
    #--Retrieving the motions
    if (csv_file_path == 'data/bodyvocab/annotation_vocab.csv'):
        motions = pd.read_csv(csv_file_path,sep=",",usecols=(3,4),skiprows=0,encoding='utf8')
                           
    else:
        motions = pd.read_csv(csv_file_path,sep=",",usecols=(3,4),skiprows=0,encoding='utf8')
    
    motions = motions.to_numpy(dtype='str').tolist()         
    for i in range(len(motions)):
       motions[i][0] = motions[i][0].split(sep=',')
       motions[i][0].append(motions[i][1])
      
    names_motions = {names[i]: np.array(motions[i][0],dtype='str') for i in range(len(names))}
    
    return names_motions
    
    
def Feedback(csv_file_path, input_video_name, toCompare_video_names , k ):
    """  
    Input:
        csv_file_path: (string) should contain the annotations and names of each video
    
        toCompare_video_names: (a list of string) that contains the names top k videos
        
        k (int) : is the number of similar videos to return.
                    in case the length of toCompare_video_names > k it selects the best k,
                    which is useful for the 'model averaging' that gives an upperbound on the performance.
                
                        
    Outputs: 
            mean F-1 score over the  K best retrieved videos (Scalar)
            Selected names (array of strings)
            
    """
    #--extracting the annotations
    names_motions = extract_motion_annotations(csv_file_path)
    input_video_motions = names_motions[input_video_name]
    
    #--array that will store the f-scores
    f_scores = [] 
    
    #--iterating over the selected videos
    unique_names = set(toCompare_video_names)
        
    for name in unique_names:
        motion = names_motions[name]
        f_scores.append( 
            f_1_score(input_video_motions, motion))
        
    #--Sorting and Selecting the K videos with largest scores
    sorted_k = [[name,score] for score,name in sorted(zip(f_scores,
                                                              list(unique_names)))][::-1]
    selected_names = [sorted_k[i][0] for i in range(k)]
    selected_scores = [sorted_k[i][1] for i in range(k)]
    
    
    return np.mean(np.array(selected_scores)), selected_names
        
#---------------------------- Clustering based approach ------------------------------
def Feedback_cluster(csv_file_path, input_video_name, toCompare_video_names):
    """  
    Input:
        csv_file_path: (string) should contain the annotations and names of each video
    
        toCompare_video_names: (a list of string) that contains the names best videos to evaluate
    
    Outputs: 
            mean F-1 score over the best retrieved videos (Scalar)
            Selected names (array of strings)
            
    """
    #--extracting the annotations
    names_motions = extract_motion_annotations(csv_file_path)
    input_video_motions = names_motions[input_video_name]
    
    #--array that will store the f-scores
    f_scores = [] 
    
    #--iterating over the selected videos
    unique_names = list(set(toCompare_video_names))
        
    for name in unique_names:
        motion = names_motions[name]
        f_scores.append( 
            f_1_score(input_video_motions, motion))
        
    #--Sorting and Selecting the K videos with largest scores
    sorted_k = [[name,score] for score,name in sorted(zip(f_scores,
                                                              list(unique_names)))][::-1]
    
    selected_names = [sorted_k[i][0] for i in range(len(unique_names))]
    selected_scores = [sorted_k[i][1] for i in range(len(unique_names))]
    
    
    return np.mean(np.array(selected_scores)), selected_names

def clustering_based_method(dataset,top_k,kernel_metric ="gak",kernel=True):
    """  
    Inputs:
        dataset (dict) : is the preproceesed dataset
        
        top_k (int) : is the desired number of retrieved videos
            the clustering tries to approach this value but doesn't guarantee it.
            
        kernel_metric (string) can be one of these:
            ‘additive_chi2’, ‘chi2’, ‘linear’, ‘poly’, ‘polynomial’, ‘rbf’,
            ‘laplacian’, ‘sigmoid’, ‘cosine’,'gak'.
            
        kernel (bool): specifies wether a kernel-based method is to be used.
    
    Output:
        clusters (2D array): assigning each element of the dataset to a cluster
        i.e. [name,assignment]
        
    """
    
    size_of_dataset = len(list(dataset.keys()))
    number_of_clusters = math.floor(size_of_dataset/top_k)
    start = time.time()
    print("The dataset has : ",size_of_dataset,
          " videos with expected number of clusters : ",number_of_clusters)
    
    #---Shapping the data
    time_series = to_time_series_dataset([i[1] for i in list(dataset.items())])
    
    #--------------Computing the clusters
    #---Using a kernel based method if kernel is True
    if kernel:
        print("\n Clustering with a kernel based method.")
        model = KernelKMeans(n_clusters= number_of_clusters, max_iter=40,
                             kernel="gak",n_jobs = -1 ,random_state=0)
        
    #---Using dtw for distance calculation is kernel is False
    else:
        print("\n Clustering with DTW as a distance metric.")
        model = TimeSeriesKMeans(n_clusters= number_of_clusters, max_iter=40, 
                             metric="dtw", n_jobs = -1,random_state=0)
    
    assignments = model.fit_predict(time_series)
    #---Printing the clustering time
    print(f"\nThe computing time is {time.time() - start} seconds.\n")
    
    #---Returning the clusters i.e. names with assignments.
    all_videos_names = list(dataset.keys())
    clusters = [[name,assign] for assign,name in  sorted(zip(assignments,all_videos_names))]
    
    return clusters
    
def retrieve_top_k_clusters(input_video_name,clusters,csv_file_path):
    """          
    Input:
        input_video_name : (string) is the name of the input video
        
        csv_file_path: (string) should contain the annotations and names of each video
    
        toCompare_video_names: (a list of string) that contains the names of the videos
    
    Outputs: 
            score (float) : mean F-1 score over the  best retrieved videos
            Selected names (array of strings)
    
    """
    #----Extracting the cluster to which the input video belongs to
    names = [ name for name,assignment in clusters]
    index = names.index(input_video_name)
    assignment = clusters[index][1]
    
    similar_videos_names = [names for names,assign in clusters if assign == assignment]
    score, _ = Feedback_cluster(csv_file_path, input_video_name, similar_videos_names)
    
    return score, similar_videos_names
        

  



    
    
    














