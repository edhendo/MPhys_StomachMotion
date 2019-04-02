# first_PCA_Attempt.py
"""
Created on Tue Nov 20 16:20:53 2018

@author: Edward Henderson
"""
import time
import numpy as np
import nibabel as nib
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

np.set_printoptions(precision=2, suppress=True)

def myPCA(data, dims_rescaled_data=9):
    """
    returns: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """
    from scipy import linalg
    data = np.array(data)
    m, n = data.shape
    # mean centre the data
    data -= data.mean(axis=0)
    # calculate the implicit covariance matrix (transpose is swapped around during the dot product multiplication)
    R = np.zeros((n,n))
    for g in range(data.shape[1]):
        R += (((data[:,g].reshape(m,1)).T).dot(data[:,g].reshape(m,1)))
        R = (1/(n-1))*R
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric, 
    # the performance gain is substantial
    evals, evecs = linalg.eigh(R)
    # sort eigenvalue in decreasing order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :dims_rescaled_data]
    # select evals too
    evals = evals[:dims_rescaled_data]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    return np.dot(data,evecs), evals, evecs
    #return evecs, evals
    
tStart = time.time()
data1 =""
np.set_printoptions(precision=4, suppress=True)

# Define which scan is the reference scan, aka the maximum exhale scan
refScanNum = 9
#------------------------------------------------------------------------------
# PANC01 has maxExhale at 9 (including the first two boxes)
# All 9 except stomach02 = 10
#------------------------------------------------------------------------------
# First extract all required warp vectors from the respective nifti images
counter = 0
for i in range(1,11):
    if True: #i != refScanNum: #Not needed right now since the reference scan is set to -1*averagewarp
        locals()["img"+str(i)] = nib.load('C:\MPhys\\Data\\Intra Patient\\Stomach\\Stomach02\\warp{0}.nii'.format(i+2))
        #locals()["img"+str(i)] = nib.load('C:\MPhys\\Nifti_Images\\Stomach_Interpolated\\Panc01\\warp{0}.nii'.format(i+2)) # plus two for the panc deformations
        locals()['hdr'+str(i)] = locals()['img'+str(i)].header
        locals()['data'+str(i)] = locals()['img'+str(i)].get_fdata()
        counter = counter + 1
        print("extracted warp vectors from DVF " + str(counter) + " out of 9")
        if counter == 10:
            print("Warning: included refScan")

# first construct the big matrix
data = np.zeros((data1.shape[0]*data1.shape[1]*data1.shape[2]*3,10)) # change to 10 if refScan included !!!
# 3 displacement values per voxel, want correlation between all

# fill the matrix V
t1 = time.time()
DVFindex = 0
for DVFnum in range(1,11):
    n = 0
    if True: #DVFnum != refScanNum: Again not needed right now since we want to include the reference scan (mean centering)       
        for x1 in range(data1.shape[0]):
            for y1 in range(data1.shape[1]):
                for z1 in range(data1.shape[2]):
                    for j in range(3):
                        data[n][DVFindex] = locals()['data'+str(DVFnum)][x1][y1][z1][0][j]
                        n = n + 1
        DVFindex = DVFindex + 1
print("Matrix filled in: " + str(np.round(time.time()-t1)) + " seconds")
    
pca, evals, evecs = myPCA(data, dims_rescaled_data=9)
    
    
    