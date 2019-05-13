# bodyBurn.py
"""
Created on Fri Dec 14 12:50:31 2018

@author: Edward Henderson
"""

import numpy as np
import nibabel as nib

#body = nib.load("C:\MPhys\\Nifti_Images\\panc02BodyBurn.nii").get_fdata()
#bodyMinus = nib.load("C:\MPhys\\Nifti_Images\\panc02Body-0.5Burn.nii").get_fdata()

lung104 = np.load("C:\MPhys\\Data\\PCA results\pcaLung104fixedNew.npy")
burn = np.array(nib.load("C:\MPhys\\lung104EXTERNALBurn.nii").get_fdata())

#lung104 = np.rot90(np.rot90(lung104,1,(0,1)),1,(0,2))

np.rot90(burn,1,(1,2))[14][3][:] = np.rot90(burn,1,(1,2))[14][4][:] # case specific for lung 104 auto-delineation error
lung104Prime = np.delete(lung104,0,1)

noBlung104result = np.zeros((49,88,51,9,3))

for i in range(9):
    for j in range(3):
        noBlung104result[:,:,:,i,j] = np.multiply(burn,lung104Prime[:,:,:,i,j])

np.save("C:\MPhys\\Data\\PCA results\\pcaLung104fixed_noBNew.npy",noBlung104result)
