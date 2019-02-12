# singleComponent picture gen
"""
Created on Sat Dec 29 12:55:03 2018

@author: Edward Henderson
"""

import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from skimage import color

sliceChoice = 14
comp = 1# remember this is componentChoice - 1
patientNum = 104

plt.figure()
pca104 = np.load('C:\MPhys\\Data\\singleCompPCA\\Rpca104_result_cube.npy')
rotated = np.rot90(pca104,1,(1,2))
plt.title('Patient ' + str(patientNum) + ' single-component PCA Weighted components 2-9')#
plt.xlabel("L-R voxel number")
plt.ylabel("Craniocaudal voxel number")
plt.imshow((2*rotated[sliceChoice,:,:,comp])+rotated[sliceChoice,:,:,comp+1]+rotated[sliceChoice,:,:,comp+2]+rotated[sliceChoice,:,:,comp+3]+rotated[sliceChoice,:,:,comp+4]+rotated[sliceChoice,:,:,comp+5]+rotated[sliceChoice,:,:,comp+6]+rotated[sliceChoice,:,:,comp+7], cmap = 'nipy_spectral')
plt.colorbar(orientation = 'horizontal')
plt.savefig('C:\MPhys\\reportImages\\singCompRpca104\\weightedComps2to9.png')
plt.show()

#rotated[sliceChoice,:,:,comp+1]+rotated[sliceChoice,:,:,comp+2]+rotated[sliceChoice,:,:,comp+3]