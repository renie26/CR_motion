This folder contatins scripts to pre-process data for feature constrcution, including original keypoint extraction using OpenPose and PoseNet, data transformation such as time-series calculation and harmonization for different purposes, [core] feature expansion and dimensionality reduction to constrcut the final features.

# Keypoint Extraction

In order to extract the data regarding the motion of different body parts we used the TensorFlow model PoseNet, which does a pose estimation of an image. For each frame in each video, we use PoseNet
to obtain the position of the different body parts, and we subsequently compute the joint angles.

## OpenPose (Body25)

### Workflow
```model``` = ```body25```

### ```data```


## PosNet

### Workflow
```models``` =  ```posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite```
In the directory ```feature_extraction```, we can find ```feature_extraction_rawcode.py```. This file contains the functions which allow to extract the feature on the different body parts, including the coordinates and the angles of the joints. The file ```keypoints_sideView.py``` is the implementation of the code that returns the text files of the extracted features of the different bodyparts for the set ```posecuts_sideView```, based on the PoseNet model of tensorflow.

Here is an explanation of the body parts indices uses to extract the angles.
    
    ### Matching keypoints indices in the output of PoseNet
    0. Left shoulder to right shoulder (5-6)
    1. Left shoulder to left elbow (5-7)
    2. Right shoulder to right elbow (6-8)
    3. Left elbow to left wrist (7-9)
    4. Right elbow to right wrist (8-10)
    5. Left hip to right hip (11-12)
    6. Left shoulder to left hip (5-11)
    7. Right shoulder to right hip (6-12)
    8. Left hip to left knee (11-13)
    9. Right hip to right knee (12-14)
    10. Left knee to left ankle (13-15)
    11.  Right knee to right ankle (14-16)
    
Important note: 2 videos from the frontview dataset couldn't be processed by the Posenet model. So we did our analyses on 38 videos instead of 40. This shows that in some configuration, the PoseNet model may not succeed in doing the pose estimation with great confidence. Nevertheless, we were able to tune and evaluate the model on these 38 videos.

### ```data```
The ```data``` directory contains the extracted features for each of the three dataset.  There is a dedicated directory for each of the three video databases, where all the features obtained from the methods in section ```feature_extraction``` are stored. For reason of confidentiality, and for explanatory purposes, only the features extracted for the videos in ```posecuts_sideviews``` are shown here.


# Pre-processing


# Feature Constrcution 
```feature_processing.py``` contains all the functions necessary to process the raw data extracted in the previous step. In particular, the function ```dataset_preprocessing``` performs the following operations:
- standardization of the data (using the library ```sklearn```)
- feature expansion, which includes cosine, sine, square root and polynomial expansion of degree 3.
- computation of the angular velocity and acceleration, which are important to analyse the motion that occurs in a video. The velocities are calculated as the difference in angles between frames. The accelerations are found as differences of velocities.
- dimensionality reduction, using PCA (from the library ```sklearn```)