"""
Implemented by Y.H EPFL 2020
"""
#python ../search_t.py face00103135 "1"

import sys
model = sys.argv[2] # 1: upper body, 2: lower body, 3: body stem, 4: right side, 5: left side

from dtwcluster_t import key,filenames,trajectories

searchterm = sys.argv[1] + '_keypoints.csv'
simterm=[]

print("Query Video:" + searchterm)

fidx = 0
for file in filenames:
    if file == searchterm:
        break
    else:
        fidx +=1    

searchIdx = fidx+1

#searchIdx = int(sys.argv[1])
#searchterm = filenames[searchIdx-1]

print(searchIdx) # prints var1
for key, _ in trajectories.items():
    if searchIdx in key:
        for index in key:
            simterm.append(filenames[index-1].rsplit("_", 1)[0]+'.mp4')

print('Clustering Results:')
print(simterm)



'''
for key, _ in trajectories.items():
    if searchIdx in key:
        for index in key:
            print("sim_v:"+filenames[index-1])

'''