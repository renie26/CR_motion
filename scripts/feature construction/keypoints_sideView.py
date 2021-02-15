import argparse
import os
import time
import pickle
from Keypoints_extraction.Feature_extraction_RawCode import get_video_features

parts_to_compare = [(5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (11, 12), (5, 11), (6, 12), (11, 13), (12, 14), (13, 15),
                    (14, 16)]

# Initiate the parser
parser = argparse.ArgumentParser()
parser.add_argument("-M", "--model_path", help="path to the tflite model",
                    default="models/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite")
parser.add_argument('--video_dir', type=str, default='./data/posecuts_sideview' ) 

# Read arguments from the command line
args = parser.parse_args()





if __name__ == '__main__':
    
    starting_time = time.time()
    
    dataset_angles = dict()
    dataset_coords = dict()
    
    #Getting the paths of all videos 
    all_videos_paths = [
        f.path for f in os.scandir(args.video_dir) if f.is_file() and f.path.endswith(('.mov'))]
        
    print("\n","Starting computation....")
    for index,path in enumerate(all_videos_paths):
        
        angles, coordinates = get_video_features(args.model_path, path)
        video_name = os.path.basename(path)
        
        print("\n",f"...Dealing with video {video_name} \n")
        
        dataset_angles.update({video_name:angles})
        dataset_coords.update({video_name:coordinates})
    
    with open('./data/posecuts_sideview/outputFile_angles.txt','wb') as outputFile :
        pickle.dump(dataset_angles,outputFile)
    
    with open('./data/posecuts_sideview/outputFile_coords.txt','wb') as outputFile :
        pickle.dump(dataset_coords,outputFile)
        
    print("Time to compute:", round(time.time() - starting_time),"seconds\n")   
    #print(dataset.keys())
    
