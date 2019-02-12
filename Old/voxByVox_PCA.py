# voxByVox_PCA.py
"""
Created on Tue Nov 27 11:14:53 2018

@author: Edward Henderson
"""
import numpy as np
import nibabel as nib
from sklearn.decomposition import PCA

data1 =""
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

def cart3sph(x,y,z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r

# First extract all required warp vectors from the respective nifti images
counter = 0
for i in range(1,11):
    if i != 7: #Not needed right now since the reference scan is set to -1*averagewarp
        locals()["img"+str(i)] = nib.load('C:\MPhys\\Nifti_Images\\cropped\\104non\\warp{0}.nii'.format(i))
        locals()['hdr'+str(i)] = locals()['img'+str(i)].header
        locals()['data'+str(i)] = locals()['img'+str(i)].get_fdata()
        #locals()["u"+str(i)],locals()["v"+str(i)],locals()["w"+str(i)] = deVectorize(locals()["data"+str(i)],np.shape(locals()["data"+str(i)])[0],np.shape(locals()["data"+str(i)])[1],np.shape(locals()["data"+str(i)])[2])
        counter = counter + 1
        print("extracted warp vectors from DVF " + str(counter) + " out of 9")

# CARTESIAN

# perform voxel by voxel Principle Components Analysis
pca = PCA(n_components=1)
# iterate through each voxel with for loops on each image dimension x, y, z
pcaCartResult = np.zeros((data1.shape[0],data1.shape[1],data1.shape[2],3))
pcaCartResult1 = np.zeros((data1.shape[0],data1.shape[1],data1.shape[2],3))
counter2 = 0
for x in range(data1.shape[0]):
    for y in range(data1.shape[1]):
        for z in range(data1.shape[2]):
            # then work with a currentVoxel numpy array of the parameters
            currentVox = np.zeros((9,3))
            warpNum = 0
            for i in range(1,11):
                if i != 7:    
                    # currently setup to extract cartesian components
                    currentVox[warpNum][0] = locals()['data'+str(i)][x][y][z][0][0]
                    currentVox[warpNum][1] = locals()['data'+str(i)][x][y][z][0][1]
                    currentVox[warpNum][2] = locals()['data'+str(i)][x][y][z][0][2]
                    warpNum = warpNum + 1
            # perform PCA step
            evecs, evals = myPCA(currentVox,1)
            pcaCartResult[x][y][z][:] = np.dot(evecs,evals)                  
            pca.fit_transform(currentVox)
            pcaCartResult1[x][y][z][:] = pca.components_      
    counter2 = counter2 + (data1.shape[1]*data1.shape[2])
    print("Voxels processed: " + str(counter2) + " of " + str(2*data1.shape[0]*data1.shape[1]*data1.shape[2]))
            
# Spherical POLAR
# voxel by voxel PCA
pcaPolResult = np.zeros((data1.shape[0],data1.shape[1],data1.shape[2],3))
pcaPolResult1 = np.zeros((data1.shape[0],data1.shape[1],data1.shape[2],3))

for x in range(data1.shape[0]):
    for y in range(data1.shape[1]):
        for z in range(data1.shape[2]):
            # then work with a currentVoxel numpy array of the parameters
            currentVox = np.zeros((9,3))
            warpNum = 0
            for i in range(1,11):
                if i != 7:    
                    az,el,r = cart3sph(locals()['data'+str(i)][x][y][z][0][0],locals()['data'+str(i)][x][y][z][0][1],locals()['data'+str(i)][x][y][z][0][2])
                    currentVox[warpNum][0] = az
                    currentVox[warpNum][1] = el
                    currentVox[warpNum][2] = r
                    warpNum = warpNum + 1
            # perform PCA step
            evecs, evals = myPCA(currentVox,1)
            pcaPolResult[x][y][z][:] = np.dot(evecs,evals)                  
            pca.fit_transform(currentVox)
            pcaPolResult1[x][y][z][:] = pca.components_      
    counter2 = counter2 + (data1.shape[1]*data1.shape[2])
    print("Voxels processed: " + str(counter2) + " of " + str(2*data1.shape[0]*data1.shape[1]*data1.shape[2]))

# Now save the resultant PCA data as .npy arrays

np.save('C:\MPhys\\Data\\voxByVox\\pca104CartResult.npy', pcaCartResult)
np.save('C:\MPhys\\Data\\voxByVox\\pca104PolResult.npy', pcaPolResult)
np.save('C:\MPhys\\Data\\voxByVox\\pca104CartResult1.npy', pcaCartResult1)
np.save('C:\MPhys\\Data\\voxByVox\\pca104PolResult1.npy', pcaPolResult1)
# accessed through np.load(path)

    













    
    