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
stomach = nib.load('C:\MPhys\\Data\\Intra Patient\\Pancreas\\niftyregPanc01StomachCrop\\stomach.nii')
stomachHdr = stomach.header;
stomachData = stomach.get_fdata();
'''
stomach_PRV = nib.load('C:\MPhys\\Data\\Intra Patient\\Pancreas\\niftyregPanc01StomachCrop\\stomach_PRVMask.nii');
stomach_PRVHdr = stomach_PRV.header;
stomach_PRVData = stomach_PRV.get_fdata();
'''
# numpy array conversion
stom = np.array(stomachData);
#stomPRV= np.array(stomach_PRVData);

# Use marching cubes to obtain the surface mesh of the stomach/stomach PRV delineations
# input 3d volume - masking data form WM
verts, faces, normals, values = measure.marching_cubes_lewiner(stom, 50)

#------------------ save vertex and face coordinates into txt files --------------------------------------------------------------
np.savetxt('C:\MPhys\\Data\\Intra Patient\\Pancreas\\3D Vis\\stomachVerts01.txt',verts, fmt = '%0.6f')
# create new array for faces - needs to have 4 components (-1 as fourth)
facesnew = np.ndarray(shape = (faces.shape[0],4))
for i in range(faces.shape[0]):
    for j in range(3):
        facesnew[i][j] = int(faces[i][j])
    facesnew[i][3] = int(-1)
    
np.savetxt('C:\MPhys\\Data\\Intra Patient\\Pancreas\\3D Vis\\stomachFaces01.txt',facesnew.astype(int), fmt = '%0.0f')
#--------------------------------------------------------------------------------------------------------------------------------

# Display resulting triangular mesh
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

# Fancy indexing: `verts[faces]` to generate a collection of triangles
mesh = Poly3DCollection(verts[faces])
mesh.set_edgecolor('k')
ax.add_collection3d(mesh)

#set axis labels
ax.set_xlabel("x-axis: L-R?")
ax.set_ylabel("y-axis: A-P?")
ax.set_zlabel("z-axis: C-C?")

#plot axis limits based on mesh dimensions
ax.set_xlim(verts[:,0].min() - 2, verts[:,0].max() + 2)
ax.set_ylim(verts[:,1].min() - 2, verts[:,1].max() + 2)  
ax.set_zlim(verts[:,2].min() - 2, verts[:,2].max() + 2)  

plt.tight_layout()
plt.show()

#----------------- assign PCA colour values --------------------------------------------------------------------------------------
#find the PCA vector values that correspond with mesh vertices
#put the PCA values that match the rounded vertex values into an array
colours = np.ndarray(shape = (verts.shape[0],3))
#round vertex numbers to nearest int
verts_round = (np.around(verts)).astype(int)

for x in range(verts.shape[0]):
    for l in range(3):
        colours[x,l] = pca_result_cube[verts_round[x,0],verts_round[x,1],verts_round[x,2],0,l] 
                                        
#write into a text file
np.savetxt('C:\MPhys\\Data\\Intra Patient\\Pancreas\\3D Vis\\stomachColours01.txt', colours, fmt = '%0.6f')

