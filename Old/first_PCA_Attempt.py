# first_PCA_Attempt.py
"""
Created on Tue Nov 20 16:20:53 2018

@author: Edward Henderson
"""
import numpy as np
import nibabel as nib
from sklearn.decomposition import PCA

np.set_printoptions(precision=2, suppress=True)

def myPCA(data, dims_rescaled_data=2):
    """
    returns: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """
    from scipy import linalg
    data = np.array(data)
    m, n = data.shape
    # mean centre the data
    data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = np.cov(data, rowvar=False)
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
    #return np.dot(evecs.T, data.T).T, evals, evecs
    return evecs, evals
    
def deVectorize(dataVec, iRange, jRange, kRange):
    u = np.zeros((iRange, jRange, kRange))
    v = np.zeros((iRange, jRange, kRange))
    w = np.zeros((iRange, jRange, kRange))
    for i in range(iRange):
        for j in range(jRange):
            for k in range(kRange):
                u[i][j][k] = dataVec[i][j][k][0][0]
                v[i][j][k] = dataVec[i][j][k][0][1]
                w[i][j][k] = dataVec[i][j][k][0][2]
    return u,v,w
counter = 0
for i in range(1,11):
    if i != 6:
        locals()["img"+str(i)] = nib.load('C:\MPhys\\Nifti_Images\\101ExMidPFix\\warp{0}.nii'.format(i))
        locals()['hdr'+str(i)] = locals()['img'+str(i)].header
        locals()['data'+str(i)] = locals()['img'+str(i)].get_fdata()
        locals()["u"+str(i)],locals()["v"+str(i)],locals()["w"+str(i)] = deVectorize(locals()["data"+str(i)],np.shape(locals()["data"+str(i)])[0],np.shape(locals()["data"+str(i)])[1],np.shape(locals()["data"+str(i)])[2])
        counter = counter + 1
        print("extracted warp vectors from DVF " + str(counter) + " out of 9")

testVec = [[u1[50][50][50],v1[50][50][50],w1[50][50][50]]]
for i in [10,9,8,7,5,4,3,2]:
    testVec = np.concatenate((testVec,[[locals()["u"+str(i)][50][50][50],locals()["v"+str(i)][50][50][50],locals()["w"+str(i)][50][50][50]]]),axis=0)
    
pca = PCA(n_components=2)

pca.fit_transform(testVec)
    
    
    
    