# server_tSNE.py
"""
Created on Thu Mar 21 15:34:55 2019

@author: Edward Henderson
"""
import time
import numpy as np
import nibabel as nib
from MulticoreTSNE import MulticoreTSNE as multiTSNE

toggle = False; ### Toggle to pre-load the tsne data cube if already filled

def cart3sph(x,y,z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r

tStart = time.time()
data1 =""
np.set_printoptions(precision=4, suppress=True)

# Define which scan is the reference scan, aka the maximum exhale scan
#------------------------------------------------------------------------------
# PANC01 has maxExhale at 9
# Stomach04 has maxExhale at 9
maxExhale = 9
#------------------------------------------------------------------------------
# First extract all required warp vectors from the respective nifti images
counter = 0

for i in range(1,11):
    locals()["img"+str(i)] = nib.load('D:\data\\Pancreas\\MPhys\\Nifti_Images\\Stomach\\Stomach04\\warp{0}.nii'.format(i+2)) # plus two for the panc deformations
    locals()['hdr'+str(i)] = locals()['img'+str(i)].header
    locals()['data'+str(i)] = locals()['img'+str(i)].get_fdata()
    counter = counter + 1
    print("extracted warp vectors from DVF " + str(counter) + " out of 9")
    if counter == 10:
        print("Warning: included refScan")

# fill the matrix for t-SNE analysis
tMatFill = time.time()
if (toggle):
    dataMatrix = np.load('D:\data\\Pancreas\\MPhys\\TSNE results\\Stomach04data.npy');
else:
    dataMatrix = np.zeros((data1.shape[0]*data1.shape[1]*data1.shape[2],3*10))
    m = 0
    xIndex = 0
    yIndex = 0
    zIndex = 0
    
    for x in range(data1.shape[0]):
        for y in range(data1.shape[1]):
            for z in range(data1.shape[2]):
                eleIndex = 0
                for DVFnum in range(1,11):
                    az, el, r = cart3sph(locals()['data'+str(DVFnum)][x][y][z][0][0],locals()['data'+str(DVFnum)][x][y][z][0][1],locals()['data'+str(DVFnum)][x][y][z][0][2])
                    for j in range(3):
                        dataMatrix[m][eleIndex] = locals()['data'+str(DVFnum)][x][y][z][0][j]
                        eleIndex += 1
                    #dataMatrix[m][eleIndex] = az
                    #eleIndex += 1
                    #dataMatrix[m][eleIndex] = el
                    #eleIndex += 1
                    #dataMatrix[m][eleIndex] = r
                    #eleIndex += 1
                    #dataMatrix[m][eleIndex] = x    # also give it the original voxel poisitions?!
                    #eleIndex += 1
                    #dataMatrix[m][eleIndex] = y
                    #eleIndex += 1
                    #dataMatrix[m][eleIndex] = z
                    #eleIndex += 1
                m = m + 1
    np.save('D:\data\\Pancreas\\MPhys\\TSNE results\\Stomach04data.npy', dataMatrix);
    
print("Filled huge matrix in: " + str(np.round(time.time()-tMatFill)) + " seconds")

# perform voxel-by-voxel t-SNE analysis
tTSNE = time.time()
tsneResult = multiTSNE(n_components=2, n_iter=1000, learning_rate=200).fit_transform(dataMatrix);
print("t-SNE completed in:" + str(np.round(time.time()-tTSNE)) + " seconds")
np.save('D:\data\\Pancreas\\MPhys\\TSNE results\\Stomach04TSNEresult.npy', tsneResult);

###############################################################################
# --> Now reassemble the data cube to align with the stomach model
voxelNum = 0
tsne_result_cube = np.zeros((data1.shape[0],data1.shape[1],data1.shape[2]));

for a in range(data1.shape[0]):
    for b in range(data1.shape[1]):
        for c in range(data1.shape[2]):
            tsne_result_cube[a][b][c] = tsneResult[voxelNum];
            voxelNum += 1;

np.save('C:\MPhys\\Data\\TSNE results\\Stomach04TSNEresultcube.npy', tsne_result_cube);

print("Program completed in: " + str(np.round(time.time()-tStart)) + " seconds");
