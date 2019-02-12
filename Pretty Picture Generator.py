# pretty picture generator
"""
Created on Thu Dec 13 11:44:04 2018

@author: Eleanor
"""

import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from skimage import color

sliceChoice = 14
comp = 0 # remember this is componentChoice + 1
patientNum = 104
pca104 = np.load('C:\MPhys\\Data\\PCA results\\pcaLung104fixed_noBNew.npy')
rotated = np.rot90(pca104,1,(1,2))
'''
plt.figure(1)
pca104 = np.load('C:\MPhys\\Data\\PCA results\\pcaLung104fixed_noB.npy')
rotated = np.rot90(pca104,1,(1,2))
plt.imshow((rotated[sliceChoice,:,:,comp]),cmap = 'nipy_spectral')
plt.title('wSum components')
plt.xlabel("L-R voxel number")
plt.ylabel("Craniocaudal voxel number")
plt.savefig('C:\MPhys\\reportImages\\run2\\wSumComp.png')
plt.show()


+rotated[sliceChoice,:,:,comp+1]+rotated[sliceChoice,:,:,comp+2]+rotated[sliceChoice,:,:,comp+3]+rotated[sliceChoice,:,:,comp+4]+rotated[sliceChoice,:,:,comp+5]+rotated[sliceChoice,:,:,comp+6]+rotated[sliceChoice,:,:,comp+7]+rotated[sliceChoice,:,:,comp+8]
'''
plt.figure(1)
plt.imshow(rotated[sliceChoice,:,:,comp],clim = (0.0,1))
plt.title('Patient ' + str(patientNum) + ' First Principal Component')
plt.xlabel("L-R voxel number")
plt.ylabel("Craniocaudal voxel number")
#plt.savefig('C:\MPhys\\reportImages\\fixed\\noBpca104_P'+str(comp+1)+'\\allcomponents.png')
plt.show()

plt.figure(2)
plt.title('Patient ' + str(patientNum) + ' x component of PC '+str(comp+1))
plt.xlabel("L-R voxel number")
plt.ylabel("Craniocaudal voxel number")
plt.imshow(rotated[sliceChoice,:,:,comp,0], cmap = 'nipy_spectral')
plt.colorbar(orientation = 'horizontal')
#plt.savefig('C:\MPhys\\reportImages\\fixed\\noBpca104_P'+str(comp+1)+'\\xcomponents.png')
plt.show()

plt.figure(3)
plt.title('Patient ' + str(patientNum) + ' y component of PC '+str(comp+1))
plt.xlabel("L-R voxel number")
plt.ylabel("Craniocaudal voxel number")
plt.imshow(rotated[sliceChoice,:,:,comp,1],cmap = 'nipy_spectral')
plt.colorbar(orientation = 'horizontal')
#plt.savefig('C:\MPhys\\reportImages\\fixed\\noBpca104_P'+str(comp+1)+'\\ycomponents.png')
plt.show()

plt.figure(4)
plt.title('Patient ' + str(patientNum) + ' z component of PC '+str(comp+1))
plt.xlabel("L-R voxel number")
plt.ylabel("Craniocaudal voxel number")
plt.imshow(rotated[sliceChoice,:,:,comp,2],cmap = 'nipy_spectral')
plt.colorbar(orientation = 'horizontal')
#plt.savefig('C:\MPhys\\reportImages\\fixed\\noBpca104_P'+str(comp+1)+'\\zcomponents.png')
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