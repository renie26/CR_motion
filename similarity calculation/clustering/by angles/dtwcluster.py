#import random
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
NUM_OF_jointAngles = 2
MIN_LEN_OF_TRAJECTORY = 16
MAX_LEN_OF_TRAJECTORY = 40
'''

model = int(sys.argv[2])
searchterm = sys.argv[1] + '_keypoints.csv'

# 1: upper body, 2: lower body, 3: left side, 4: right side, 5: legs and arms
model_col_set = [['#head_JA', '#r_shoulder_JA', '#l_shoulder_JA', '#r_upleg_JA','#l_upleg_JA','#r_elbow_JA', '#l_elbow_JA', '#upbody_JA'],
                ['#waist_JA','#r_body_JA', '#l_body_JA', '#r_knee_JA', '#l_knee_JA'],
                ['#head_JA', '#l_shoulder_JA','#l_upleg_JA','#l_elbow_JA','#l_body_JA','#l_knee_JA'],
                ["#head_JA", '#r_shoulder_JA', '#r_upleg_JA','#r_elbow_JA', '#upbody_JA','#r_body_JA','#r_knee_JA'],
                ['#r_upleg_JA','#l_upleg_JA','#r_elbow_JA', '#l_elbow_JA','#r_knee_JA','#l_knee_JA']]
th_set = [300, 300, 300, 300, 300]

model_col = []
THRESHOLD = 300 #face

model_col = model_col_set[model -1]
THRESHOLD = th_set[model -1]

print("Selected Model:" + model_col)

root = os.getcwd()

dirs = os.listdir(root)

#data
jAngleSet = {}

filenames =[]

filenames.append(searchterm)

tempJointAngle = pd.read_csv(searchterm)
tempJointAngle.columns = ['#head_JA', '#r_shoulder_JA', '#l_shoulder_JA', '#r_upleg_JA','#l_upleg_JA','#r_elbow_JA', '#l_elbow_JA', '#upbody_JA', '#waist_JA','#r_body_JA', '#l_body_JA', '#r_knee_JA', '#l_knee_JA']
X = tempJointAngle[model_col] 
X = np.array(X) # save to array
jAngleSet[(0,)] = X

i = 1
for file in dirs:
    if file !=searchterm:
        if file.endswith((".csv")):
            tempJointAngle = pd.read_csv(file)
            filenames.append(file)
            #'#head_JA', '#r_shoulder_JA', '#l_shoulder_JA', '#r_upleg_JA','#l_upleg_JA', 
            #'#r_elbow_JA', '#l_elbow_JA', '#upbody_JA', '#waist_JA','#r_body_JA', '#l_body_JA', '#r_knee_JA', '#l_knee_JA'],
            tempJointAngle.columns = ['#head_JA', '#r_shoulder_JA', '#l_shoulder_JA', '#r_upleg_JA','#l_upleg_JA','#r_elbow_JA', '#l_elbow_JA', '#upbody_JA', '#waist_JA','#r_body_JA', '#l_body_JA', '#r_knee_JA', '#l_knee_JA']
            X = tempJointAngle[model_col] 
            X = np.array(X) # save to array
            jAngleSet[(i,)] = X
            i = i+1
#print(jAngleSet)

'''
for item in range(NUM_OF_jointAngles):
   length = random.choice(list(range(MIN_LEN_OF_TRAJECTORY, MAX_LEN_OF_TRAJECTORY + 1)))
   tempJointAngle = np.random.randint(low=-100, high=100, size=int(length / 4)).astype(float) / 100
   
   oldScale = np.arange(0, int(length / 4))
   interpolationFunction = interpolate.interp1d(oldScale, tempJointAngle)
   
   newScale = np.linspace(0, int(length / 4) - 1, length)
   tempJointAngle = interpolationFunction(newScale)
   
   jAngleSet[(str(item),)] = [tempJointAngle]
print(jAngleSet)
'''


jointAngles = deepcopy(jAngleSet)
#print(jointAngles)

distanceMatrixDictionary = {}

iteration = 1
while True:
    distanceMatrix = np.empty((len(jointAngles), len(jointAngles),))
    distanceMatrix[:] = np.nan
    
    for index1, (filter1, angleTerm1) in enumerate(jointAngles.items()):
        tempArray = []
        
        for index2, (filter2, angleTerm2) in enumerate(jointAngles.items()):
            
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
                for subItem1 in angleTerm1:
                    
                    for subItem2 in angleTerm2:
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
    
    filter1 = list(jointAngles.keys())[minIndex[0]]
    filter2 = list(jointAngles.keys())[minIndex[1]]
    
    angleTerm1 = jointAngles.get(filter1)
    angleTerm2 = jointAngles.get(filter2)
    
    unionFilter = filter1 + filter2

    sorted(unionFilter)
    if len(angleTerm1) > len(angleTerm2):
        angleGroup = angleTerm2.copy()
        angleGroup += angleTerm1[:len(angleTerm2)]
    else:
        angleGroup = angleTerm1.copy()
        angleGroup += angleTerm2[:len(angleTerm1)] 

    #angleGroup = angleTerm1 + angleTerm2

    jointAngles = {key: value for key, value in jointAngles.items()
                    if all(value not in unionFilter for value in key)}
    
    distanceMatrixDictionary = {key: value for key, value in distanceMatrixDictionary.items()
                                if all(value not in unionFilter for value in key)}
    
    jointAngles[unionFilter] = angleGroup
    
    print(iteration, 'finished!')
    iteration += 1
    
    if len(list(jointAngles.keys())) == 1:
        break


for key, _ in jointAngles.items():
    print(key)
    
'''
for key, _ in jointAngles.items():
    if 1 in key:
        for index in key:
            print("sim_v:"+filenames[index-1])
'''
'''
for key, value in jointAngles.items():
    
    if len(key) == 1:
        continue
    
    figure = make_subplots(rows=1, cols=1)
    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(len(value))]
    
    for index, subValue in enumerate(value):
        
        figure.add_trace(go.Scatter(x=list(range(0, len(subValue))), y=subValue,
                                    mode='lines', marker_color=colors[index], line = dict(width=4), line_shape='spline'), row=1, col=1,
                        )
    
    figure.update_layout(showlegend=False, height=600, width=900)
    figure.show()
'''