# firstVisualisationAttempt.py
"""
Created on Tue Nov  6 16:27:42 2018

@author: Edward Henderson
"""

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from sklearn.decomposition import PCA


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

np.set_printoptions(precision=2, suppress=True)
imgLoad = nib.load('C:\MPhys\\Nifti_Images\\101\\averageVecs.nii')
hdr = imgLoad.header
data = imgLoad.get_fdata()

fig = plt.figure(figsize=(20,18))
ax = fig.add_subplot(111, projection='3d')


x, y, z = np.meshgrid(np.arange(0, 83, 1),
                      np.arange(0, 83, 1),
                      np.arange(0, 74, 1))

u,v,w = deVectorize(data,83,83,74)

ax.quiver(x, y, z, u, v, w, length=0.1, pivot='middle')
#plt.show()
#fig.savefig('C:\MPhys\\Python_Images\\101\\averageVecs3.png')