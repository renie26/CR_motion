import random
from copy import deepcopy
import os
#from scipy import interpolate
import numpy as np
import pandas as pd
from dtaidistance import dtw
import sys

#import matplotlib.pyplot as plt
#from _plotly_future_ import v4_subplots
#import plotly.graph_objects as go
#from plotly.subplots import make_subplots

'''
NUM_OF_TRAJECTORIES = 2
MIN_LEN_OF_TRAJECTORY = 16
MAX_LEN_OF_TRAJECTORY = 40
'''

model = int(sys.argv[2])
searchterm = sys.argv[1] + '_keypoints.csv'

# 1: upper body, 2: lower body, 3: left side, 4: right side, 5: legs and arms
model_col_set = [['#head', '#r_shoulder', '#l_shoulder', '#r_upleg','#l_upleg','#r_elbow', '#l_elbow', '#upbody'],
                ['#r_waist','#l_waist', '#r_knee', '#l_knee'],
                ['#head', '#l_shoulder', '#l_upleg', '#l_elbow', '#upbody','#l_waist', '#l_knee'],
                ["#head", '#r_shoulder', '#r_upleg', '#r_elbow', '#upbody','#r_waist', '#r_knee'],
                ['#r_upleg','#l_upleg','#r_elbow', '#l_elbow', '#r_knee', '#l_knee']]
th_set = [200, 200, 120, 100, 100]

model_col = []
THRESHOLD = 100 #face

model_col = model_col_set[model -1]
THRESHOLD = th_set[model -1]

print("Selected Model:" + model_col)

root = os.getcwd()

dirs = os.listdir(root)

#data
trajectoriesSet = {}

filenames =[]

filenames.append(searchterm)

tempTrajectory = pd.read_csv(searchterm)
tempTrajectory.columns = ['#head', '#r_shoulder', '#l_shoulder', '#r_upleg','#l_upleg','#r_elbow', '#l_elbow', '#upbody', '#r_waist','#l_waist', '#r_knee', '#l_knee']
X = tempTrajectory[model_col] 
X = np.array(X) # save to array
trajectoriesSet[(0,)] = X

i = 1
for file in dirs:
    if file !=searchterm:
	    if file.endswith((".csv")):
	        tempTrajectory = pd.read_csv(file)
	        filenames.append(file)
	        #'#head', '#r_shoulder', '#l_shoulder', '#r_upleg','#l_upleg', 
	        #'#r_elbow', '#l_elbow', '#upbody', '#r_waist','#l_waist', '#r_knee', '#l_knee'],
	        tempTrajectory.columns = ['#head', '#r_shoulder', '#l_shoulder', '#r_upleg','#l_upleg','#r_elbow', '#l_elbow', '#upbody', '#r_waist','#l_waist', '#r_knee', '#l_knee']
	        X = tempTrajectory[model_col] 
	        X = np.array(X) # save to array
	        trajectoriesSet[(i,)] = X
	        i = i+1
#print(trajectoriesSet)

'''
for item in range(NUM_OF_TRAJECTORIES):
   length = random.choice(list(range(MIN_LEN_OF_TRAJECTORY, MAX_LEN_OF_TRAJECTORY + 1)))
   tempTrajectory = np.random.randint(low=-100, high=100, size=int(length / 4)).astype(float) / 100
   
   oldScale = np.arange(0, int(length / 4))
   interpolationFunction = interpolate.interp1d(oldScale, tempTrajectory)
   
   newScale = np.linspace(0, int(length / 4) - 1, length)
   tempTrajectory = interpolationFunction(newScale)
   
   trajectoriesSet[(str(item),)] = [tempTrajectory]
print(trajectoriesSet)
'''


trajectories = deepcopy(trajectoriesSet)
#print(trajectories)

distanceMatrixDictionary = {}

iteration = 1
while True:
    distanceMatrix = np.empty((len(trajectories), len(trajectories),))
    distanceMatrix[:] = np.nan
    
    for index1, (filter1, trajectory1) in enumerate(trajectories.items()):
        tempArray = []
        
        for index2, (filter2, trajectory2) in enumerate(trajectories.items()):
            
            if index1 > index2:
                continue
            
            elif index1 == index2:
                continue
            
            else:
                unionFilter = filter1 + filter2
                sorted(unionFilter)
                
                if unionFilter in distanceMatrixDictionary.keys():
                    distanceMatrix[index1][index2] = distanceMatrixDictionary.get(unionFilter)
                    
                    continue
                
                metric = []
                for subItem1 in trajectory1:
                    
                    for subItem2 in trajectory2:
                        metric.append(dtw.distance(subItem1, subItem2, psi=1))
                
                metric = max(metric)
                
                distanceMatrix[index1][index2] = metric
                distanceMatrixDictionary[unionFilter] = metric
    
    minValue = np.min(list(distanceMatrixDictionary.values()))
    
    print(minValue)
    if minValue > THRESHOLD:
        break
    
    minIndices = np.where(distanceMatrix == minValue)
    minIndices = list(zip(minIndices[0], minIndices[1]))
    
    minIndex = minIndices[0]
    
    filter1 = list(trajectories.keys())[minIndex[0]]
    filter2 = list(trajectories.keys())[minIndex[1]]
    
    trajectory1 = trajectories.get(filter1)
    trajectory2 = trajectories.get(filter2)
    
    unionFilter = filter1 + filter2

    sorted(unionFilter)

    if len(trajectory1) > len(trajectory2):
        trajectoryGroup = trajectory2.copy()
        trajectoryGroup += trajectory1[:len(trajectory2)]
    else:
        trajectoryGroup = trajectory1.copy()
        trajectoryGroup += trajectory2[:len(trajectory1)] 

    #trajectoryGroup = trajectory1 + trajectory2

    trajectories = {key: value for key, value in trajectories.items()
                    if all(value not in unionFilter for value in key)}
    
    distanceMatrixDictionary = {key: value for key, value in distanceMatrixDictionary.items()
                                if all(value not in unionFilter for value in key)}
    
    trajectories[unionFilter] = trajectoryGroup
    
    print(iteration, 'finished!')
    iteration += 1
    
    if len(list(trajectories.keys())) == 1:
        break


for key, _ in trajectories.items():
    print(key)

'''
for key, value in trajectories.items():
    
    if len(key) == 1:
        continue
    
    figure = make_subplots(rows=1, cols=1)
    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(len(value))]
    
    for index, subValue in enumerate(value):
        
        figure.addrace(go.Scatter(x=list(range(0, len(subValue))), y=subValue,
                                    mode='lines', marker_color=colors[index], line = dict(width=4), line_shape='spline'), row=1, col=1,
                        )
    
    figure.update_layout(showlegend=False, height=600, width=900)
    figure.show()
'''