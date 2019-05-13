# stomach_isosurface.py
"""
Created on Thu Mar  7 14:58:10 2019

@author: Edward Henderson
"""

import time
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

t1 = time.time();
np.set_printoptions(precision=4, suppress=True);
#pca_result_cube = np.load('C:\MPhys\\Data\\PCA results\\niftyregPanc01StomachCropPCAcube.npy');
pca_result_cube = np.load('C:\MPhys\\Data\\Intra Patient\\Pancreas\\PCA\\niftyregPanc01StomachCropPCAcube.npy')

# Read in the delineation nifti files using nibabel
#stomach = nib.load('C:\MPhys\\Nifti_Images\\niftyregPanc01StomachCrop\\stomachMask.nii');
stomach = nib.load('C:\MPhys\Data\Intra Patient\Pancreas\niftyregPanc01StomachCrop\\stomachMask.nii')
stomachHdr = stomach.header;
stomachData = stomach.get_fdata();
stomach_PRV = nib.load('C:\MPhys\Data\Intra Patient\Pancreas\niftyregPanc01StomachCrop\\stomach_PRVMask.nii');
stomach_PRVHdr = stomach_PRV.header;
stomach_PRVData = stomach_PRV.get_fdata();

# numpy array conversion
stom = np.array(stomachData);
stomPRV= np.array(stomach_PRVData);

# Use marching cubes to obtain the surface mesh of the stomach/stomach PRV delineations
# input 3d volume - masking data form WM
verts, faces, normals, values = measure.marching_cubes_lewiner(, 50)

# Display resulting triangular mesh
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Fancy indexing: `verts[faces]` to generate a collection of triangles
mesh = Poly3DCollection(verts[faces])
mesh.set_edgecolor('k')
ax.add_collection3d(mesh)