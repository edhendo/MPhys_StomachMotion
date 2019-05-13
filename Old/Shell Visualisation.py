# Shell Visulisation - 1st Attempt
"""
Created on Tue Feb 26 16:15:32 2019

@author: Eleanor
"""
import time
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

t1 = time.time();
np.set_printoptions(precision=4, suppress=True);
# Eleanor path
pca_result_cube = np.load('C:\\MPhys\\Data\\Intra Patient\\Pancreas\\PCA\\niftyregPanc01StomachCropPCAcube.npy');
# Ed path
#pca_result_cube = np.load('C:\\MPhys\\Data\\PCA results\\niftyregPanc01StomachCropPCAcube.npy');
# Read in the delineation nifti files using nibabel
stomach = nib.load('C:\\MPhys\\Data\\Intra Patient\\Pancreas\\niftyregPanc01StomachCrop\\stomachMask.nii');
#stomach = nib.load('C:\\MPhys\\Nifti_Images\\niftyregPanc01StomachCrop\\stomachMask.nii')
stomachHdr = stomach.header;
stomachData = stomach.get_fdata();
stomach_PRV = nib.load('C:\\MPhys\\Data\\Intra Patient\\Pancreas\\niftyregPanc01StomachCrop\\stomach_PRVMask.nii');
#stomac_PRV = nib.load('C:\\MPhys\\Nifti_Images\\niftyregPanc01StomachCrop\\stomach_PRVMask.nii')
stomach_PRVHdr = stomach_PRV.header;
stomach_PRVData = stomach_PRV.get_fdata();
# numpy array conversion
stom = np.array(stomachData);
stomPRV= np.array(stomach_PRVData);
# calculate outer shell
outerShell = stomPRV - stom;

#multiply the masks and the PCA cube together to obtain a shell
#need to ensure that the multiplication is voxel-by-voxel
for component in range(9):
    for i in range(pca_result_cube.shape[0]):
        for j in range(pca_result_cube.shape[1]):
            for k in range(pca_result_cube.shape[2]):
                for xyz in range(3):
                    pca_result_cube[i,j,k,component,xyz] = pca_result_cube[i,j,k,component,xyz]*outerShell[i,j,k];

print("Program completed in: " + str(np.round(time.time()-t1)) + " seconds");
#save shell PCA 
np.save('C:\\MPhys\\Data\\Intra Patient\\Pancreas\\PCA\\pcaShellPanc01',pca_result_cube)
#np.save('C:\\MPhys\\Data\\PCA results\\niftyregPanc01StomachCropPCAshell.npy',pca_result_cube)

#visualise the shell in 3D - via mesh/voxels
#re-name shell cube
pca_shell = pca_result_cube

#determine x,y,z coordinates and u,v,w coordinates
'''
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
'''
fig = plt.figure()
ax = fig.gca(projection='3d')

X,Y,Z = np.meshgrid(np.arange(0,pca_shell[:,0,0,0,0].size,1),
        np.arange(0,pca_shell[0,:,0,0,0].size,1), 
        np.arange(0,pca_shell[0,0,:,0,0].size,1))

U,V,W = pca_shell[X,Y,Z,0,0],pca_shell[X,Y,Z,0,1],pca_shell[X,Y,Z,0,2]
#create colour list

colour = (pca_shell[X,Y,Z,0,0],pca_shell[X,Y,Z,0,1],pca_shell[X,Y,Z,0,2],1.0)
colour_filtered = filter(lambda x: (x=(0.0,0.0,0.0,1)), colour)
colour2 = colour_filtered + colour_filtered
ax.quiver(X,Y,Z,U,V,W, colors = colour2)

plt.show()