# CR Video Search Tool based on Motion Similarity
EDDH Semester Project #1, Yumeng H

# Introduction

This project is based on a database of videos of rituals. The scope of this project is to find a method to retrieve the _k_ videos that are the most similar to a given one, in the database of the video. 

The dataset that we have is divided in three databases:
- ```posecuts_frontview``` consists of 40 videos showing the front view of a person performing determined actions in the framework of certain rituals.
- ```posecuts_sideview```, is relatively similar to the first dataset in the fact that it also consists of 40 videos. However, this time they show the side view of a person performing different actions.
- ```bodyvocab``` comprises 133 videos of varied types of actions. The videos in this dataset are more heterogeneous  and complex in their nature than in the other two, as the persons may perform different types of actions from different views, and other objects may appear. Certain videos also include actions performed by two separate persons.

For the first two databases, we have also been given some textual annotations for each video to be used for assessing the model.

Due to confidentiality reasons, the demo used in this repository solely focuses on the dataset ```posecuts_sideview```.

# Required External Libraries

In order to carry out this project, the following external libraries have been used:
- ```TensorFlow```
- ```scikit-learn```
- ```tslearn```
- ```scipy```

# Dependencies
natsort
numpy
pandas
dtaidistance
copy
random
sys
os
math
time
pickle