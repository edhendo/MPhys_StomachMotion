# Full-image PCA - 3M rows
"""
Created on Tue Dec 11 13:47:27 2018

@author: Ed Henderson
"""

import time
import numpy as np
import nibabel as nib
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

tStart = time.time()
data1 =""
np.set_printoptions(precision=4, suppress=True)

# Define which scan is the reference scan, aka the maximum exhale scan
refScanNum = 9
#------------------------------------------------------------------------------
# PANC01 has maxExhale at 9 (including the first two boxes)
# 7 for 104
#------------------------------------------------------------------------------
# First extract all required warp vectors from the respective nifti images
counter = 0
for i in range(1,11):
    if True: #i != refScanNum: #Not needed right now since the reference scan is set to -1*averagewarp
        locals()["img"+str(i)] = nib.load('C:\MPhys\\Nifti_Images\\niftyregPanc01StomachCrop\\warp{0}.nii'.format(i+2)) # plus two for the panc deformations
        locals()['hdr'+str(i)] = locals()['img'+str(i)].header
        locals()['data'+str(i)] = locals()['img'+str(i)].get_fdata()
        counter = counter + 1
        print("extracted warp vectors from DVF " + str(counter) + " out of 9")
        if counter == 10:
            print("Warning: included refScan")

# Perform PCA over the whole image with 3M rows, 10 columns
# voxel| DVF1 | DVF2  | ...  | DVF10   One DVF is ref scan - 10 columns
#   1  | x11  |  x12  | x1.  | x110
#   1  | y11  |  y12  | y1.  | y110
#   1  | z11  |  z12  | z1.  | z110
#   2  | x21  |  x22  | x2.  | x210
#   2  | y21  |  y22  | y2.  | y210
#   2  | z21  |  z22  | z2.  | z210
#  ... | ..1  |  ..2  | ...  | ..10
#   M  | xM1  |  xM2  | xM.  | xM10
#   M  | yM1  |  yM2  | yM.  | yM10
#   M  | zM1  |  zM2  | zM.  | zM10

# M is total number of voxels eg M = 49*89*51 dependent on cropping
# exclude the reference scan (only use 9 DVFs)
# patient 104 has 49x89x51
# panc02 has 38*56*38

# first construct the big matrix
data = np.zeros((data1.shape[0]*data1.shape[1]*data1.shape[2]*3,10)) # change to 10 if refScan included !!!
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
                    for j in range(3):
                        data[n][DVFindex] = locals()['data'+str(DVFnum)][x1][y1][z1][0][j]
                        n = n + 1
        DVFindex = DVFindex + 1
print("Matrix filled in: " + str(np.round(time.time()-t1)) + " seconds")
# PCA step
t2 = time.time()
pca = PCA(n_components=9) # set to N-1 components for correlated matrix PCA
pca_result = pca.fit_transform(data)
print("PCA completed in: " + str(np.round(time.time()-t2)) + " seconds")
print("Explained variation per principal component: {}".format(pca.explained_variance_ratio_))
# sklearn PCA uses the implicit covariance (correlated) matrix PCA method outlined in Sohn 2005

# Now read the principle components from the PCA back into a data cube for slice by slice visualisation
t3 = time.time()
min_max_scaler = MinMaxScaler()
for PCAcompIndex in range(9):
    locals()['component' + str(PCAcompIndex + 1)] = np.zeros((data1.shape[0] * data1.shape[1] * data1.shape[2]*3,1))
    for voxelNum in range(data1.shape[0] * data1.shape[1] * data1.shape[2]*3):
        locals()['component' + str(PCAcompIndex + 1)][voxelNum] = pca_result[voxelNum][PCAcompIndex]
    locals()['scaledComponent' + str(PCAcompIndex + 1)] = min_max_scaler.fit_transform((locals()['component' + str(PCAcompIndex + 1)]).reshape(-1,1))

voxelNum = 0
pca_result_cube = np.zeros((data1.shape[0],data1.shape[1],data1.shape[2],9,3))
for PCAcompIndex2 in range(9):
    voxelNum = 0
    for x2 in range(data1.shape[0]):
        for y2 in range(data1.shape[1]):
            for z2 in range(data1.shape[2]):
                    for k in range(3):
                        pca_result_cube[x2][y2][z2][PCAcompIndex2][k] = locals()['scaledComponent' + str(PCAcompIndex2 + 1)][voxelNum]
                        voxelNum = voxelNum + 1

print("Data reshaped in: " + str(np.round(time.time()-t3)) + " seconds")

# end
print("Program completed in: " + str(np.round(time.time()-tStart)) + " seconds")

# Now save the resultant PCA data as .npy arrays

np.save('C:\MPhys\\Data\\PCA results\\niftyregPanc01StomachCropPCAcube.npy', pca_result_cube)
# accessed through np.load(path)

#produce graph of variance ratios
pcomponents = np.linspace(1,9,9)
plt.plot(pcomponents, pca.explained_variance_ratio_,'o-', markersize = 5, clip_on = False)
plt.title('Percentage Variance - Panc01 Stomach Crop',fontsize = 16)
plt.xlabel('Principal Component', fontsize = 16)
plt.ylabel('Percentage of total variance', fontsize = 16)
plt.ylim(0,1.0)
plt.xlim(1,9)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.grid(True)
plt.savefig('C:\MPhys\\Python_Images\\niftyregPanc01StomachCrop\\PCvariance.png')