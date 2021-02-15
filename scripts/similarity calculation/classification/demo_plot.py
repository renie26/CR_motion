"""  
This Demo will execute the similarity calculation based on
 dynamic time wrapping and  a Radial basis function.
It will output a graph that displays the performance of the model.

"""

import pickle
from similarity_calculation import retrieve_top_k
from feature_processing import dataset_preprocessing
import time
import numpy as np
import matplotlib.pyplot as plt

#--files paths
#stored_angles = 'data/frontview/outputFile_angles.txt'
#csv_annotations = 'data/frontview/annotation_front.csv'
stored_angles = 'data/sideview/outputFile_angles.txt'
csv_annotations = 'data/sideview/annotation_side.csv'

#--opening file
with open(stored_angles,'rb') as file:
    dataset_angles = pickle.load(file) 

#--Demo
all_videos_names = list(dataset_angles.keys())
scores = []
times = []
below_35_percent = dict()  #This will store the videos with low performace
results_scores = dict()
results_time = dict()

for k in [3,6,9]:
    print(f'\n Printing the score of the Top {k} similar videos to : \n')
    for name in all_videos_names[:5]:
        input_video_name =  name 
        start = time.time()
    
        #----------------  Preprocessing --------------------------------------------------------------------
        Raw_input_video = dataset_angles[input_video_name]

        print("Query Video: " + input_video_name)
        processed_input_video, preprocessed_dataset = dataset_preprocessing(Raw_input_video,dataset_angles)
        
        del Raw_input_video
        
        #----------------- Retrieving ------------------------------------------------------------------------
        score, selected_names = retrieve_top_k(processed_input_video,input_video_name,
                                               preprocessed_dataset, k , csv_annotations,model_averaging = True )
        
        #print("\n\n",name,f" --> F-1 score {round(score)}% \n\n",
         #     "Computing time {:.2f} [s] ".format(time.time()-start))
         
        print("\n Top-K Results:",selected_names,"\n")
        scores.append(score)
        times.append(time.time()-start)
        
        #---Storing the video with score below 35
        if(score < 35):
            below_35_percent.update({name:score})
            
    print('\n\n','Median score',np.median(np.array(scores)),'\n',
          'median time',np.median(np.array(times)),
          '\n Mean score',np.mean(np.array(scores)))
    
    #---Storing results for plotting
    results_scores.update({f'Top{k}':np.array(scores)})
    results_time.update({f'Top {k}':np.array(times)})
    
    #---Printing the videos with low performance
    print(f'\n Printing the {len(list(below_35_percent.keys()))} low performing videos: ')
    for key,value in below_35_percent.items():
        print('{} --> {:.2f}%'.format(key,value))

#------------ Plotting --------------------
fig, ax = plt.subplots(2,1,figsize=(10,10))
ax[0].boxplot(results_scores.values())
ax[0].set_title(f'Scores, Sideview , {len(all_videos_names)} videos, Model averaging') #Sideview FrontView
ax[0].set(ylabel = 'F-1 score in %' )
ax[0].set_xticklabels(results_scores.keys())

ax[1].boxplot(results_time.values())
ax[1].set_title(f'Computing time, Sideview, {len(all_videos_names)} videos, Model averaging')
ax[1].set(ylabel='Time in seconds')
ax[1].set_xticklabels(results_time.keys())

plt.show()



    
    