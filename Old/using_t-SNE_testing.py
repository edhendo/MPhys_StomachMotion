# using_t-SNE_testing.py
"""
Created on Tue Nov 27 12:55:47 2018

@author: Edward Henderson

Old t-SNE script used for testing

"""

import numpy as np
import nibabel as nib
from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE as multiTSNE
import time
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

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
refScanNum = 7
#------------------------------------------------------------------------------
# SETUP FOR PANC02, MAXEXHALE = 8, CAREFUL WITH WARP NUMBERS
# PANC01 has maxExhale at 7
# lung 104 at 7
#------------------------------------------------------------------------------
# First extract all required warp vectors from the respective nifti images
counter = 0
for i in range(1,11):
    if i != refScanNum: #Not needed right now since the reference scan is set to -1*averagewarp
        locals()["img"+str(i)] = nib.load('C:\MPhys\\Nifti_Images\\cropped\\104non\\warp{0}.nii'.format(i)) # plus two for the panc deformations
        #locals()["img"+str(i)] = nib.load('D:\data\\Pancreas\\MPhys\\Nifti_Images\\lung104non\\warp{0}.nii'.format(i)) # plus two for the panc deformations
        locals()['hdr'+str(i)] = locals()['img'+str(i)].header
        locals()['data'+str(i)] = locals()['img'+str(i)].get_fdata()
        counter = counter + 1
        print("extracted warp vectors from DVF " + str(counter) + " out of 9")
        if counter == 10:
            print("Warning: included refScan")

# fill the matrix for t-SNE analysis

data = np.zeros((data1.shape[0]*data1.shape[1]*data1.shape[2],7*10))

t1 = time.time()
m = 0
xIndex = 0
yIndex = 0
zIndex = 0
for x in range(data1.shape[0]):
    for y in range(data1.shape[1]):
        for z in range(data1.shape[2]):
            eleIndex = 0
            for DVFnum in range(1,11):
                if DVFnum != refScanNum:
                    az, el, r = cart3sph(locals()['data'+str(DVFnum)][x][y][z][0][0],locals()['data'+str(DVFnum)][x][y][z][0][1],locals()['data'+str(DVFnum)][x][y][z][0][2])
                    for j in range(3):
                        data[m][eleIndex] = locals()['data'+str(DVFnum)][x][y][z][0][j]
                        eleIndex = eleIndex + 1
                    #data[m][eleIndex] = az
                    #eleIndex = eleIndex + 1
                    #data[m][eleIndex] = el
                    #eleIndex = eleIndex + 1
                    data[m][eleIndex] = r
                    eleIndex = eleIndex + 1
                    data[m][eleIndex] = x    # also give it the original voxel poisitions?!
                    eleIndex = eleIndex + 1
                    data[m][eleIndex] = y
                    eleIndex = eleIndex + 1
                    data[m][eleIndex] = z
                    eleIndex = eleIndex + 1
            m = m + 1
print("Filled huge matrix in: " + str(np.round(time.time()-t1)) + " seconds")

# perform voxel-by-voxel t-SNE analysis
t2 = time.time()
reduced_data = TSNE(n_components=3, n_iter=1500, learning_rate=175).fit_transform(data)
print("t-SNE completed in:" + str(np.round(time.time()-t2)) + " seconds")
np.save('C:\MPhys\\Data\\PCA results\\lung104_TSNEresultWPosMPol3.npy', reduced_data)

t3 = time.time()
reduced_data2 = TSNE(n_components=2, n_iter=1500, learning_rate=175).fit_transform(data)
print("t-SNE 2 completed in:" + str(np.round(time.time()-t3)) + " seconds")
np.save('C:\MPhys\\Data\\PCA results\\lung104_TSNEresultWPosMPol2.npy', reduced_data2)

t4 = time.time()
reduced_data3 = TSNE(n_components=2, n_iter=1500, learning_rate=225).fit_transform(data)
print("t-SNE 3 completed in:" + str(np.round(time.time()-t4)) + " seconds")
np.save('C:\MPhys\\Data\\PCA results\\lung104_TSNEresultWPosMPolhighRate.npy', reduced_data3)

t5 = time.time()
reduced_data4 = multiTSNE(n_components=3, n_iter=1500, learning_rate=175).fit_transform(data)
print("t-SNE 4 completed in:" + str(np.round(time.time()-t5)) + " seconds")
np.save('C:\MPhys\\Data\\PCA results\\lung104_multiTSNEresultWPosMPol3.npy', reduced_data4)

t6 = time.time()
reduced_data5 = multiTSNE(n_components=2, n_iter=1500, learning_rate=175).fit_transform(data)
print("t-SNE 4 completed in:" + str(np.round(time.time()-t6)) + " seconds")
np.save('C:\MPhys\\Data\\PCA results\\lung104_multiTSNEresultWPosMPol2.npy', reduced_data5)

#np.save('D:\data\\Pancreas\\MPhys\\lung104_multiTSNEresult.npy', reduced_data)

'''
plt.figure()
plt.scatter(reduced_data[:,0],reduced_data[:,1],marker='.',s=0.25)
plt.xlabel("t-SNE Component 1", fontsize = "20")
plt.ylabel("t-SNE Component 2", fontsize = "20")
'''
print("Program completed in:" + str(np.round(time.time()-tStart)) + " seconds")

'''
fig2 = plt.figure()
ax = fig2.add_subplot(111, projection='3d')
ax.scatter(reduced_data[:,0],reduced_data[:,1],reduced_data[:,2])
ax.set_xlabel("component 1")
ax.set_ylabel("component 2")
ax.set_zlabel("component 3")
'''
