import json
import os
import glob
import csv
from natsort import natsort
import re
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt



def main(path, root):

    os.chdir(path) 


    def joint_angle(point1, point2, point3):
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

                p3_x = json_object['people'][0]['pose_keypoints_2d'][point_x(point3)]
                p3_y = json_object['people'][0]['pose_keypoints_2d'][point_y(point3)]

                u = np.array([p1_x - p2_x, p1_y - p2_y])
                v = np.array([p3_x - p2_x, p3_y - p2_y])

                i = np.inner(u, v)
                n = LA.norm(u) * LA.norm(v)

                if n == 0:
                        a = 0
                else:
                        c = i / n
                        a = np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0)))

        return round(a, 2) #a


    head_JA = ['#head_JA']
    r_shoulder_JA = ['#r_shoulder_JA']
    l_shoulder_JA = ['#l_shoulder_JA']
    r_upleg_JA = ['#r_upleg_JA']
    l_upleg_JA = ['#l_upleg_JA']
    r_elbow_JA = ['#r_elbow_JA']
    l_elbow_JA = ['#l_elbow_JA']
    upbody_JA = ['#upbody_JA']
    waist_JA = ['#waist_JA']
    r_body_JA = ['#r_body_JA']
    l_body_JA = ['#l_body_JA']
    r_knee_JA = ['#r_knee_JA']
    l_knee_JA = ['#l_knee_JA']

    for json_file in natsort(glob.glob('*_keypoints.json')):
        with open(json_file) as f:
            json_object = json.load(f)
            json_object_length = len(json_object['people'])

            head_JA.append(joint_angle(point1 = 17, point2 = 0, point3 = 1))
            r_shoulder_JA.append(joint_angle(point1 = 17, point2 = 1, point3 = 2))
            l_shoulder_JA.append(joint_angle(point1 = 18, point2 = 1, point3 = 5))

            r_upleg_JA.append(joint_angle(point1 = 1, point2 = 2, point3 = 3))
            l_upleg_JA.append(joint_angle(point1 = 1, point2 = 5, point3 = 6))
            r_elbow_JA.append(joint_angle(point1 = 2, point2 = 3, point3 = 4))
            l_elbow_JA.append(joint_angle(point1 = 5, point2 = 6, point3 = 7))

            upbody_JA.append(joint_angle(point1 = 2, point2 = 1, point3 = 8))
            waist_JA.append(joint_angle(point1 = 1, point2 = 8, point3 = 9))
            r_body_JA.append(joint_angle(point1 = 2, point2 = 9, point3 = 10))
            l_body_JA.append(joint_angle(point1 = 5, point2 = 12, point3 = 13))

            r_knee_JA.append(joint_angle(point1 = 9, point2 = 10, point3 = 11))
            l_knee_JA.append(joint_angle(point1 = 12, point2 = 13, point3 = 14))

    jsonpath = root+'/json'
    os.chdir(jsonpath)
    csvname = json_file.split("_",1)[0]
    csvname = csvname + '_keypoints.csv'
    rows = zip(head_JA,r_shoulder_JA,l_shoulder_JA,r_upleg_JA,l_upleg_JA,r_elbow_JA,l_elbow_JA,upbody_JA,waist_JA,r_body_JA,l_body_JA,r_knee_JA,l_knee_JA)
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

