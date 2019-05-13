# panc pretty picture generator
"""
Created on Sat Dec 29 19:55:22 2018

@author: Edward Henderson
"""

import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from skimage import color

sliceChoice = 14
comp = 1 # remember this is componentChoice - 1
patientNum = 104
'''
plt.figure(1)
pca104 = np.load('C:\MPhys\\Data\\PCA results\\pca104_result_cube_3Mscalednon_noB.npy')
rotated = np.rot90(pca104,1,(1,2))
plt.imshow(rotated[sliceChoice,:,:,comp+1]+rotated[sliceChoice,:,:,comp+2]+rotated[sliceChoice,:,:,comp+3],cmap = 'nipy_spectral')
plt.title('wSum components')
plt.xlabel("L-R voxel number")
plt.ylabel("Craniocaudal voxel number")
plt.savefig('C:\MPhys\\reportImages\\run2\\wSumComp.png')
plt.show()
'''

plt.figure(1)
pca104 = np.load('C:\MPhys\\Data\\PCA results\\panc02_result_cube_3M.npy')
rotated = np.rot90(pca104,1,(1,2))
plt.imshow(rotated[sliceChoice,:,:,comp],clim = (0.0,0.5))
plt.title('panc02 First Principal Component')
plt.xlabel("L-R voxel number")
plt.ylabel("Craniocaudal voxel number")
plt.savefig('C:\MPhys\\reportImages\\panc02\\allcomponents.png')
plt.show()

plt.figure(2)
plt.title('panc02 x component of PC '+str(comp+1))
plt.xlabel("L-R voxel number")
plt.ylabel("Craniocaudal voxel number")
plt.imshow(rotated[sliceChoice,:,:,comp,0], cmap = 'nipy_spectral')
plt.colorbar(orientation = 'horizontal')
plt.savefig('C:\MPhys\\reportImages\\panc02\\xcomponents.png')
plt.show()

plt.figure(3)
plt.title('panc02 y component of PC '+str(comp+1))
plt.xlabel("L-R voxel number")
plt.ylabel("Craniocaudal voxel number")
plt.imshow(rotated[sliceChoice,:,:,comp,1],cmap = 'nipy_spectral')
plt.colorbar(orientation = 'horizontal')
plt.savefig('C:\MPhys\\reportImages\\panc02\\ycomponents.png')
plt.show()

plt.figure(4)
plt.title('panc02 z component of PC '+str(comp+1))
plt.xlabel("L-R voxel number")
plt.ylabel("Craniocaudal voxel number")
plt.imshow(rotated[sliceChoice,:,:,comp,2],cmap = 'nipy_spectral')
plt.colorbar(orientation = 'horizontal')
plt.savefig('C:\MPhys\\reportImages\\panc02\\zcomponents.png')
plt.show()

'''
save images with a picture of the lungs (reg image beacuse scaling is the same)

pca104 = np.load('C:\MPhys\\Data\\PCA results\\pca104_result_cube_3Mscaled.npy')
rotated = np.rot90(pca104,1,(1,2))
plt.figure(1)
plt.scatter(rotated[:,:,:,2,0],rotated[:,:,:,2,1], s = 0.1, marker = ".", c = 'm')
plt.figure(2)
plt.scatter(rotated[:,:,:,2,0],rotated[:,:,:,2,2], s = 0.1, marker = ".", c = 'c')
plt.figure(3)
plt.scatter(rotated[:,:,:,2,1],rotated[:,:,:,2,2], s = 0.1, marker = ".", c = 'g')
'''