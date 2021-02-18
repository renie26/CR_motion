import json
import os
import glob
import csv
from natsort import natsort
import re
import numpy as np
from numpy import linalg as LA
#import matplotlib.pyplot as plt



def main(path, root):

    os.chdir(path) 


    def avg_pos(point1, point2):
        def point_x(number):
                return number * 3

        def point_y(number):
                return (number * 3) + 1

        if json_object_length == 0:
                a = 0

        else:
                p1_x = json_object['people'][0]['pose_keypoints_2d'][point_x(point1)]
                p1_y = json_object['people'][0]['pose_keypoints_2d'][point_y(point1)]

                p2_x = json_object['people'][0]['pose_keypoints_2d'][point_x(point2)]
                p2_y = json_object['people'][0]['pose_keypoints_2d'][point_y(point2)]

                u = (p1_x + p2_x)/2
                v = (p1_y + p2_y)/2

                c = LA.norm(u) * LA.norm(v)

                a = np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0)))

        return round(a, 2) #a


    head_t = ['#head']
    r_shoulder_t = ['#r_shoulder']
    l_shoulder_t = ['#l_shoulder']
    r_upleg_t = ['#r_upleg']
    l_upleg_t = ['#l_upleg']
    r_elbow_t = ['#r_elbow']
    l_elbow_t = ['#l_elbow']
    upbody_t = ['#upbody']
    r_waist_t = ['#r_waist']
    l_waist_t = ['#rl_waist']
    r_knee_t = ['#r_knee']
    l_knee_t = ['#l_knee']

    for json_file in natsort(glob.glob('*_keypoints.json')):
        with open(json_file) as f:
            json_object = json.load(f)
            json_object_length = len(json_object['people'])

            head_t.append(avg_pos(point1 = 0, point2 = 1))
            r_shoulder_t.append(avg_pos(point1 = 2, point2 = 2))
            l_shoulder_t.append(avg_pos(point1 = 5, point2 = 5))

            r_upleg_t.append(avg_pos(point1 = 2, point2 = 3))
            l_upleg_t.append(avg_pos(point1 = 5, point2 = 6))
            r_elbow_t.append(avg_pos(point1 = 3, point2 = 4))
            l_elbow_t.append(avg_pos(point1 = 6, point2 = 7))

            upbody_t.append(avg_pos(point1 = 1, point2 = 8))
            r_waist_t.append(avg_pos(point1 = 9, point2 = 9))
            l_waist_t.append(avg_pos(point1 = 12, point2 = 12))

            r_knee_t.append(avg_pos(point1 = 10, point2 = 11))
            l_knee_t.append(avg_pos(point1 = 13, point2 = 14))

    jsonpath = root+'/jpose'
    os.chdir(jsonpath)
    csvname = json_file.split("_",1)[0]
    csvname = csvname + '_keypoints.csv'
    rows = zip(head_t,r_shoulder_t,l_shoulder_t,r_upleg_t,l_upleg_t,r_elbow_t,l_elbow_t,upbody_t,r_waist_t,l_waist_t,r_knee_t,l_knee_t)
    with open(csvname, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        for row in rows:
            writer.writerow(row)

    os.chdir(path) 


root = os.getcwd()

dirs = os.listdir(root)
for file in dirs:
    if file.endswith((".mp4")):
        path = root +'/'+file
        main(path, root)
        print path
        os.chdir(root)

