import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

'''
#search by term
def dataset_preprocessing_specified(input_video, dirs):    
    dataset_preprocessed = dict()
    tempJointAngle = pd.read_csv(input_video)
    tempJointAngle.columns = ['#head_JA', '#r_shoulder_JA', '#l_shoulder_JA', '#r_upleg_JA','#l_upleg_JA','#r_elbow_JA', '#l_elbow_JA', '#upbody_JA', '#waist_JA','#r_body_JA', '#l_body_JA', '#r_knee_JA', '#l_knee_JA']
    X_in = tempJointAngle[model_col] 
    X_in = np.array(X_in) # save to array
    X_in = featureExpansion_angles(X_in,convertToRadian=False)
    X_in = dimensionality_reduction(X_in)
    dataset_preprocessed.update({0: X_in})

    i = 1
    for file in dirs:
        if file.endswith((".csv")):
            tempJointAngle = pd.read_csv(file)
            filenames.append(file)
            tempJointAngle.columns = ['#head_JA', '#r_shoulder_JA', '#l_shoulder_JA', '#r_upleg_JA','#l_upleg_JA','#r_elbow_JA', '#l_elbow_JA', '#upbody_JA', '#waist_JA','#r_body_JA', '#l_body_JA', '#r_knee_JA', '#l_knee_JA']
            X = tempJointAngle[model_col] 
            X = np.array(X) # save to array
            X = featureExpansion_angles(X,convertToRadian=False)
            X = dimensionality_reduction(X)
            dataset_preprocessed.update({i: X})
            i = i+1
    return X_in, dataset_preprocessed
'''

def dataset_preprocessing(raw_video,lookUp_dataset):
    """   This method preprocesses the dataset and the input video
    
            Input video :       Raw_Video
            lookUp_dataset :    Raw look up dataset
            
            Output:
                preprocessed input video (numpy.array)
                preprocessed dataset (numpy.array)
    """
    
    dataset_preprocessed = dict()
    for key,value in lookUp_dataset.items():
        dummy = np.array(value)
        dummy = featureExpansion_angles(dummy,convertToRadian=False)
        
        dummy = dimensionality_reduction(dummy)
        dataset_preprocessed.update({key: dummy})
        
    preprocessed_video = featureExpansion_angles(raw_video)
    preprocessed_video = dimensionality_reduction(preprocessed_video)
    
    return preprocessed_video , dataset_preprocessed


#---------------------------------------- Feature expansion ------------------------------------------------------------------
def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    
    poly = np.ones((len(x), 1))
    for deg in range(1, degree + 1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly


def featureExpansion_angles(video1,convertToRadian=False):
    """  This function conducts a feature expansion
        
        cosine expansion
        sine expansion
        sqrt expansion
        polynomial expansion degree 3
        Velocity and acceleration 
        
       Output:
               feature expanded array (numpy.array)
               NB: the method puts emphasis on features along the x axis
    """
    #---Ensuring that input video is a numpy.array
    if type(video1) != type(np.array([0,0])):
        sample = np.array(video1)
    else :
        sample = video1
        
    #---converting to radians if specified
    if convertToRadian:
        sample = np.radians(sample)  
        
    #---Standardization
    sample = StandardScaler().fit_transform(sample)
    
    #--- Feature expansion
    cosine_sample = np.cos(sample)
    sine_sample = np.sin(sample)
    log_sample = np.log(sample**2 + 1)
    
    sample_velocity_x = np.zeros_like(sample)
    sample_velocity_y = np.zeros_like(sample)
    sample_acceleration_x = np.zeros_like(sample)
    sample_acceleration_y = np.zeros_like(sample)
    
    #---Computing velocities along frames
    for i in range(1,sample.shape[0]):
        sample_velocity_x[i,:] = np.abs(cosine_sample[i,:] - cosine_sample[i-1,:])
        sample_velocity_y[i,:] = np.abs(sine_sample[i,:] - sine_sample[i-1,:])
        
     #---Computing accelerations along frames   
    for i in range(1,sample.shape[0]):
        sample_acceleration_x[i,:] = np.abs(sample_velocity_x[i,:] - sample_velocity_x[i-1,:])
        sample_acceleration_y[i,:] = np.abs(sample_velocity_y[i,:] - sample_velocity_y[i-1,:])
        
    #---Polynomial expansion
    poly_x = build_poly(sample_acceleration_x * sample_velocity_x ,
                        degree=3)[:,1:]
    
    #---stacking columnwise and putting more emphasis on the x axis
    output = np.hstack((cosine_sample**2,sine_sample, log_sample,
                        sample_velocity_x**2,
                        sample_velocity_y,poly_x,
                        sample_acceleration_x**2,sample_acceleration_y
                        ))

    
    return output

#--------------------------------------Dimensionality reduction  PCA -----------------------------------------------
def dimensionality_reduction(video):
    """ Implement Principal component analysis
   
        Input:  
                video: it represents the matrix (n*d) that contains all pose features in the video
                                n --> number of frames in the video
                                d --> number of features

       Outout:
                returns a matrix containing the 24 biggest components of the variance of the input
    """
    
    #pca = PCA(n_components=24)
    pca = PCA(n_components=10) #ym
    transformed_video = pca.fit_transform(video)

    return transformed_video

    
    
