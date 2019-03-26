# stomach_isosurface_TSNE2.py
"""
Created on Tue Mar 26 11:54:48 2019

@author: Edward Henderson

This program will first filter out non-stomach data-points and then perform  
the t-SNE algorithm. This is to reduce computation time and improve feature
identification performance. 

"""
import time
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
import math
from sklearn.preprocessing import MinMaxScaler
from matplotlib import cm
from MulticoreTSNE import MulticoreTSNE as multiTSNE

def cart3sph(x,y,z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r

tStart = time.time()
data1 =""
np.set_printoptions(precision=4, suppress=True)
#------------------------------------------------------------------------------
# First extract all required warp vectors from the respective nifti images
counter = 0

for i in range(1,11):
    locals()["img"+str(i)] = nib.load('C:\MPhys\\Nifti_Images\\Stomach04\\warp{0}.nii'.format(i+2)) # plus two for the panc deformations
    #locals()["img"+str(i)] = nib.load('D:\data\\Pancreas\\MPhys\\Nifti_Images\\Stomach\\Stomach04\\warp{0}.nii'.format(i+2)) # plus two for the panc deformations
    locals()['hdr'+str(i)] = locals()['img'+str(i)].header
    locals()['data'+str(i)] = locals()['img'+str(i)].get_fdata()
    counter = counter + 1
    print("extracted warp vectors from DVF " + str(counter) + " out of 9")
    if counter == 10:
        print("Warning: included refScan")

#------------------------------------------------------------------------------     
# Read in the delineation nifti files using nibabel
stomach = nib.load('C:\MPhys\\Nifti_Images\\Stomach04\\stomachMask.nii');
# stomach = nib.load('C:\MPhys\\Data\\Intra Patient\\Pancreas\\niftyregPanc01StomachCrop\\stomach.nii')
stomachHdr = stomach.header;
stomachData = stomach.get_fdata();

# numpy array conversion
#stom = np.rot90(np.rot90(np.array(stomachData),2,(0,2)),1,(1,2));
stom = np.array(stomachData);
#stomPRV= np.array(stomach_PRVData);

# Use marching cubes to obtain the surface mesh of the stomach/stomach PRV delineations
# input 3d volume - masking data form WM
verts, faces, normals, values = measure.marching_cubes_lewiner(stom, 50)

# create new array for faces - needs to have 4 components (-1 as fourth)
facesnew = np.ndarray(shape = (faces.shape[0],4))
for i in range(faces.shape[0]):
    for j in range(3):
        facesnew[i][j] = int(faces[i][j])
    facesnew[i][3] = int(-1)     

# round vertex numbers to nearest int
verts_round = (np.around(verts)).astype(int)

#------------------------------------------------------------------------------           
# Now strip out the stoamch data alone
    
for a in range(1,11):
    locals()['data'+str(i)]    
        
        
        
        
        
        
        
        
        
        
        
        
        