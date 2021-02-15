# python ../json2txt.py face00103135 "1"

import os
import numpy as np
import pandas as pd
import sys
import pickle

searchterm = sys.argv[1] + '_keypoints.csv'
model = int(sys.argv[2])

# 1: upper body, 2: lower body, 3: left side, 4: right side, 5: legs and arms
model_col_set = [['#head_JA', '#r_shoulder_JA', '#l_shoulder_JA', '#r_upleg_JA','#l_upleg_JA','#r_elbow_JA', '#l_elbow_JA', '#upbody_JA'],
                ['#waist_JA','#r_body_JA', '#l_body_JA', '#r_knee_JA', '#l_knee_JA'],
                ['#head_JA', '#l_shoulder_JA','#l_upleg_JA','#l_elbow_JA','#l_body_JA','#l_knee_JA'],
                ["#head_JA", '#r_shoulder_JA', '#r_upleg_JA','#r_elbow_JA', '#upbody_JA','#r_body_JA','#r_knee_JA'],
                ['#r_upleg_JA','#l_upleg_JA','#r_elbow_JA', '#l_elbow_JA','#r_knee_JA','#l_knee_JA']]
model_col = []
model_col = model_col_set[model -1]

root = os.getcwd()

dirs = os.listdir(root)

#data
jAngleSet = {}
dataset_angles = dict()

tempJointAngle = pd.read_csv(searchterm)
tempJointAngle.columns = ['#head_JA', '#r_shoulder_JA', '#l_shoulder_JA', '#r_upleg_JA','#l_upleg_JA','#r_elbow_JA', '#l_elbow_JA', '#upbody_JA', '#waist_JA','#r_body_JA', '#l_body_JA', '#r_knee_JA', '#l_knee_JA']
X = tempJointAngle[model_col] 
dataset_angles.update({searchterm:X})

i = 1
for file in dirs:
    if file !=searchterm:
        if file.endswith((".csv")):
            tempJointAngle = pd.read_csv(file)
            tempJointAngle.columns = ['#head_JA', '#r_shoulder_JA', '#l_shoulder_JA', '#r_upleg_JA','#l_upleg_JA','#r_elbow_JA', '#l_elbow_JA', '#upbody_JA', '#waist_JA','#r_body_JA', '#l_body_JA', '#r_knee_JA', '#l_knee_JA']
            X = tempJointAngle[model_col] 
            dataset_angles.update({file:X})
            i = i+1

with open('../outputFile_angles.txt','wb') as outputFile :
        pickle.dump(dataset_angles,outputFile)