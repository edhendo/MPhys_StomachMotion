# Initial_stomach_shell.py
"""
Created on Tue Feb 26 15:28:44 2019

@author: Edward Henderson
"""

import time
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

t1 = time.time();
np.set_printoptions(precision=4, suppress=True);
pca_result_cube = np.load('C:\MPhys\\Data\\PCA results\\niftyregPanc01StomachCropPCAcube.npy');
# Read in the delineation nifti files using nibabel
stomach = nib.load('C:\MPhys\\Nifti_Images\\niftyregPanc01StomachCrop\\stomachMask.nii');
stomachHdr = stomach.header;
stomachData = stomach.get_fdata();
stomach_PRV = nib.load('C:\MPhys\\Nifti_Images\\niftyregPanc01StomachCrop\\stomach_PRVMask.nii');
stomach_PRVHdr = stomach_PRV.header;
stomach_PRVData = stomach_PRV.get_fdata();
# numpy array conversion
stom = np.array(stomachData);
stomPRV= np.array(stomach_PRVData);
# calculate outer shell
outerShell = stomPRV - stom;

for component in range(9):
    for i in range(pca_result_cube.shape[0]):
        for j in range(pca_result_cube.shape[1]):
            for k in range(pca_result_cube.shape[2]):
                for xyz in range(3):
                    pca_result_cube[i,j,k,component,xyz] = pca_result_cube[i,j,k,component,xyz]*outerShell[i,j,k];

print("Program completed in: " + str(np.round(time.time()-t1)) + " seconds");                