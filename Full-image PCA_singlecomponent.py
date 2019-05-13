# stomach_magnitude_PCA.py
"""
Created on Tue Mar 12 15:33:10 2019

@author: Edward Henderson
"""

import time
import numpy as np
import nibabel as nib
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import math

def magnitude(x,y,z):
    return math.sqrt((x**2 + y**2 + z**2))

tStart = time.time()
data1 =""
np.set_printoptions(precision=4, suppress=True)

# Define which scan is the reference scan, aka the maximum exhale scan
refScanNum = 9
#------------------------------------------------------------------------------
# Stomach07 has maxExhale at 9 (including the first two boxes
# Stomach07 = 10
#------------------------------------------------------------------------------
# First extract all required warp vectors from the respective nifti images
counter = 0
for i in range(1,11):
    if True: #i != refScanNum: #Not needed right now since the reference scan is set to -1*averagewarp
        #locals()["img"+str(i)] = nib.load('C:\MPhys\\Data\\Intra Patient\\Stomach_Interpolated\\Stomach07\\warp{0}.nii'.format(i+2))
        locals()["img"+str(i)] = nib.load('C:\MPhys\\Nifti_Images\\Stomach07\\warp{0}.nii'.format(i+2)) # plus two for the panc deformations
        locals()['hdr'+str(i)] = locals()['img'+str(i)].header
        locals()['data'+str(i)] = locals()['img'+str(i)].get_fdata()
        counter = counter + 1
        print("extracted warp vectors from DVF " + str(counter) + " out of 9")
        if counter == 10:
            print("Warning: included refScan")

# first construct the big matrix
data = np.zeros((data1.shape[0]*data1.shape[1]*data1.shape[2],10)) # change to 10 if refScan included !!!
# 3 displacement values per voxel, want correlation between all

# fill the matrix V
t1 = time.time()
DVFindex = 0
for DVFnum in range(1,11):
    n = 0
    if True: #DVFnum != refScanNum: Again not needed right now since we want to include the refernence scan (mean centering)       
        for x1 in range(data1.shape[0]):
            for y1 in range(data1.shape[1]):
                for z1 in range(data1.shape[2]):
                    data[n][DVFindex] = magnitude(locals()['data'+str(DVFnum)][x1][y1][z1][0][0],locals()['data'+str(DVFnum)][x1][y1][z1][0][1],locals()['data'+str(DVFnum)][x1][y1][z1][0][2]);
                    n = n + 1
        DVFindex = DVFindex + 1
print("Matrix filled in: " + str(np.round(time.time()-t1)) + " seconds")
# PCA step
t2 = time.time()
pca = PCA(n_components=9) # set to N-1 components for correlated matrix PCA
pca_result = pca.fit_transform(data)
print("PCA completed in: " + str(np.round(time.time()-t2)) + " seconds")
print("Explained variation per principal component: {}".format(pca.explained_variance_ratio_))
#np.save('C:\MPhys\\Data\\Intra Patient\\Stomach_Interpolated\\PCA\\EVRMag_Stomach07.npy',pca.explained_variance_ratio_)
# sklearn PCA uses the implicit covariance (correlated) matrix PCA method outlined in Sohn 2005

# Now read the principle components from the PCA back into a data cube for slice by slice visualisation
t3 = time.time()
for PCAcompIndex in range(9):
    locals()['component' + str(PCAcompIndex + 1)] = np.zeros((data1.shape[0] * data1.shape[1] * data1.shape[2]))
    for voxelNum in range(data1.shape[0] * data1.shape[1] * data1.shape[2]):
        locals()['component' + str(PCAcompIndex + 1)][voxelNum] = pca_result[voxelNum][PCAcompIndex]
    

voxelNum = 0
pca_result_cube = np.zeros((data1.shape[0],data1.shape[1],data1.shape[2],9))
for PCAcompIndex2 in range(9):
    voxelNum = 0
    for x2 in range(data1.shape[0]):
        for y2 in range(data1.shape[1]):
            for z2 in range(data1.shape[2]):
                pca_result_cube[x2][y2][z2][PCAcompIndex2] = locals()['component' + str(PCAcompIndex2 + 1)][voxelNum]
                voxelNum = voxelNum + 1

print("Data reshaped in: " + str(np.round(time.time()-t3)) + " seconds")

# end
print("Program completed in: " + str(np.round(time.time()-tStart)) + " seconds")

# Now save the resultant PCA data as .npy arrays

np.save('C:\MPhys\\Data\\PCA results\\Stomach07\\PCAcube.npy', pca_result_cube)
#np.save('C:\MPhys\\Data\\Intra Patient\\Stomach_Interpolated\\PCA\\pcaMagStomach07.npy', pca_result_cube)
# accessed through np.load(path)
'''
#produce graph of variance ratios
pcomponents = np.linspace(1,9,9)
plt.plot(pcomponents, pca.explained_variance_ratio_,'o-', markersize = 5, clip_on = False)
plt.title('Percentage Variance - Stomach07 Mag',fontsize = 16)
plt.xlabel('Principal Component', fontsize = 16)
plt.ylabel('Percentage of total variance', fontsize = 16)
plt.ylim(0,1.0)
plt.xlim(1,9)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.grid(True)
#plt.savefig('C:\MPhys\\Python_Images\\niftyregStomach07StomachCrop\\magnitudePCvariance.png')
plt.savefig('C:\MPhys\\Data\\Intra Patient\\Stomach_Interpolated\\PCA Graphs and Images\\Stomach07_VarianceMag.png')
'''