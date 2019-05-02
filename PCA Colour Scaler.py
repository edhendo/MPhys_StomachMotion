# PCA Colour Scaler
"""
Created on Thu May  2 15:43:44 2019

@author: Eleanor
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

def magnitude(x,y,z):
    return math.sqrt((x**2 + y**2 + z**2))

tStart = time.time();
np.set_printoptions(precision=4, suppress=True);
# Import the PCA results for all patients
pca_result_cube1 = np.load('C:\MPhys\\Data\\Intra Patient\\Stomach_Interpolated\\PCA\\pcaPanc01.npy')
pca_result_cube2 = np.load('C:\MPhys\\Data\\Intra Patient\\Stomach_Interpolated\\PCA\\pcaStomach02.npy')
pca_result_cube4 = np.load('C:\MPhys\\Data\\Intra Patient\\Stomach_Interpolated\\PCA\\pcaStomach04.npy')
pca_result_cube5 = np.load('C:\MPhys\\Data\\Intra Patient\\Stomach_Interpolated\\PCA\\pcaStomach05.npy')
pca_result_cube6 = np.load('C:\MPhys\\Data\\Intra Patient\\Stomach_Interpolated\\PCA\\pcaStomach06.npy')
pca_result_cube7 = np.load('C:\MPhys\\Data\\Intra Patient\\Stomach_Interpolated\\PCA\\pcaStomach07.npy')
    
# flip the pca data in the same manner in order to ensure the data for each point remains consistent
pca_result_cubeLR1 = np.fliplr(pca_result_cube1)
pca_result_cubeLR2 = np.fliplr(pca_result_cube2)
pca_result_cubeLR4 = np.fliplr(pca_result_cube4)
pca_result_cubeLR5 = np.fliplr(pca_result_cube5)
pca_result_cubeLR6 = np.fliplr(pca_result_cube6)
pca_result_cubeLR7 = np.fliplr(pca_result_cube7)

# Read in the delineation nifti files using nibabel
stomach1 = nib.load('C:\MPhys\\Data\\Intra Patient\\Stomach_Interpolated\\Panc01\\stomachMask.nii')
stomachData1 = stomach1.get_fdata();
stomach2 = nib.load('C:\MPhys\\Data\\Intra Patient\\Stomach_Interpolated\\Stomach02\\stomachMask.nii')
stomachData2 = stomach2.get_fdata();
stomach4 = nib.load('C:\MPhys\\Data\\Intra Patient\\Stomach_Interpolated\\Stomach04\\stomachMask.nii')
stomachData4 = stomach4.get_fdata();
stomach5 = nib.load('C:\MPhys\\Data\\Intra Patient\\Stomach_Interpolated\\Stomach05\\stomachMask.nii')
stomachData5 = stomach5.get_fdata();
stomach6 = nib.load('C:\MPhys\\Data\\Intra Patient\\Stomach_Interpolated\\Stomach06\\stomachMask.nii')
stomachData6 = stomach6.get_fdata();
stomach7 = nib.load('C:\MPhys\\Data\\Intra Patient\\Stomach_Interpolated\\Stomach07\\stomachMask.nii')
stomachData7 = stomach7.get_fdata();

# numpy array conversion
stomLR1 = np.fliplr(stomachData1)
stomLR2 = np.fliplr(stomachData2)
stomLR4 = np.fliplr(stomachData4)
stomLR5 = np.fliplr(stomachData5)
stomLR6 = np.fliplr(stomachData6)
stomLR7 = np.fliplr(stomachData7)

# Use marching cubes to obtain the surface mesh of the stomach/stomach PRV delineations
# input 3d volume - masking data form WM
verts1, faces1, normals1, values1 = measure.marching_cubes_lewiner(stomLR1, 50)
verts2, faces2, normals2, values2 = measure.marching_cubes_lewiner(stomLR2, 50)
verts4, faces4, normals4, values4 = measure.marching_cubes_lewiner(stomLR4, 50)
verts5, faces5, normals5, values5 = measure.marching_cubes_lewiner(stomLR5, 50)
verts6, faces6, normals6, values6 = measure.marching_cubes_lewiner(stomLR6, 50)
verts7, faces7, normals7, values7 = measure.marching_cubes_lewiner(stomLR7, 50)

    
#----------------- Create all the necessary arrays --------------------------------------------------------------------------------------
#find the PCA vector values that correspond with mesh vertices
#put the PCA values that match the rounded vertex values into an array
# separate x, y and z components here
# Panc01
pca_x1 = np.ndarray(shape = (verts1.shape[0]))
pca_y1 = np.ndarray(shape = (verts1.shape[0]))
pca_z1 = np.ndarray(shape = (verts1.shape[0]))
colours_x1 = np.ndarray(shape = (verts1.shape[0],3))
colours_y1 = np.ndarray(shape = (verts1.shape[0],3))
colours_z1 = np.ndarray(shape = (verts1.shape[0],3))
#round vertex numbers to nearest int
verts_round1 = (np.around(verts1)).astype(int)
for i in range(verts1.shape[0]):
    pca_x1[i] = pca_result_cubeLR1[verts_round1[i,0],verts_round1[i,1],verts_round1[i,2],0,0];
    pca_y1[i] = pca_result_cubeLR1[verts_round1[i,0],verts_round1[i,1],verts_round1[i,2],0,1];
    pca_z1[i] = pca_result_cubeLR1[verts_round1[i,0],verts_round1[i,1],verts_round1[i,2],0,2];
    
# Stomach02    
pca_x2 = np.ndarray(shape = (verts2.shape[0]))
pca_y2 = np.ndarray(shape = (verts2.shape[0]))
pca_z2 = np.ndarray(shape = (verts2.shape[0]))
colours_x2 = np.ndarray(shape = (verts2.shape[0],3))
colours_y2 = np.ndarray(shape = (verts2.shape[0],3))
colours_z2 = np.ndarray(shape = (verts2.shape[0],3))
#round vertex numbers to nearest int
verts_round2 = (np.around(verts2)).astype(int)
for i in range(verts2.shape[0]):
    pca_x2[i] = pca_result_cubeLR2[verts_round2[i,0],verts_round2[i,1],verts_round2[i,2],0,0];
    pca_y2[i] = pca_result_cubeLR2[verts_round2[i,0],verts_round2[i,1],verts_round2[i,2],0,1];
    pca_z2[i] = pca_result_cubeLR2[verts_round2[i,0],verts_round2[i,1],verts_round2[i,2],0,2];
    
# Stomach04
pca_x4 = np.ndarray(shape = (verts4.shape[0]))
pca_y4 = np.ndarray(shape = (verts4.shape[0]))
pca_z4 = np.ndarray(shape = (verts4.shape[0]))
colours_x4 = np.ndarray(shape = (verts4.shape[0],3))
colours_y4 = np.ndarray(shape = (verts4.shape[0],3))
colours_z4 = np.ndarray(shape = (verts4.shape[0],3))
#round vertex numbers to nearest int
verts_round4 = (np.around(verts4)).astype(int)
for i in range(verts4.shape[0]):
    pca_x4[i] = pca_result_cubeLR4[verts_round4[i,0],verts_round4[i,1],verts_round4[i,2],0,0];
    pca_y4[i] = pca_result_cubeLR4[verts_round4[i,0],verts_round4[i,1],verts_round4[i,2],0,1];
    pca_z4[i] = pca_result_cubeLR4[verts_round4[i,0],verts_round4[i,1],verts_round4[i,2],0,2];
    
# Stomach05
pca_x5 = np.ndarray(shape = (verts5.shape[0]))
pca_y5 = np.ndarray(shape = (verts5.shape[0]))
pca_z5 = np.ndarray(shape = (verts5.shape[0]))
colours_x5 = np.ndarray(shape = (verts5.shape[0],3))
colours_y5 = np.ndarray(shape = (verts5.shape[0],3))
colours_z5 = np.ndarray(shape = (verts5.shape[0],3))
#round vertex numbers to nearest int
verts_round5 = (np.around(verts5)).astype(int)
for i in range(verts5.shape[0]):
    pca_x5[i] = pca_result_cubeLR5[verts_round5[i,0],verts_round5[i,1],verts_round5[i,2],0,0];
    pca_y5[i] = pca_result_cubeLR5[verts_round5[i,0],verts_round5[i,1],verts_round5[i,2],0,1];
    pca_z5[i] = pca_result_cubeLR5[verts_round5[i,0],verts_round5[i,1],verts_round5[i,2],0,2];
    
# Stomach06
pca_x6 = np.ndarray(shape = (verts6.shape[0]))
pca_y6= np.ndarray(shape = (verts6.shape[0]))
pca_z6 = np.ndarray(shape = (verts6.shape[0]))
colours_x6 = np.ndarray(shape = (verts6.shape[0],3))
colours_y6 = np.ndarray(shape = (verts6.shape[0],3))
colours_z6 = np.ndarray(shape = (verts6.shape[0],3))
#round vertex numbers to nearest int
verts_round6 = (np.around(verts6)).astype(int)
for i in range(verts6.shape[0]):
    pca_x6[i] = pca_result_cubeLR6[verts_round6[i,0],verts_round6[i,1],verts_round6[i,2],0,0];
    pca_y6[i] = pca_result_cubeLR6[verts_round6[i,0],verts_round6[i,1],verts_round6[i,2],0,1];
    pca_z6[i] = pca_result_cubeLR6[verts_round6[i,0],verts_round6[i,1],verts_round6[i,2],0,2];
    
# Stomach07
pca_x7 = np.ndarray(shape = (verts7.shape[0]))
pca_y7 = np.ndarray(shape = (verts7.shape[0]))
pca_z7 = np.ndarray(shape = (verts7.shape[0]))
colours_x7 = np.ndarray(shape = (verts7.shape[0],3))
colours_y7 = np.ndarray(shape = (verts7.shape[0],3))
colours_z7 = np.ndarray(shape = (verts7.shape[0],3))
#round vertex numbers to nearest int
verts_round7 = (np.around(verts7)).astype(int)
for i in range(verts7.shape[0]):
    pca_x7[i] = pca_result_cubeLR7[verts_round7[i,0],verts_round7[i,1],verts_round7[i,2],0,0];
    pca_y7[i] = pca_result_cubeLR7[verts_round7[i,0],verts_round7[i,1],verts_round7[i,2],0,1];
    pca_z7[i] = pca_result_cubeLR7[verts_round7[i,0],verts_round7[i,1],verts_round7[i,2],0,2];

#----------------- PCA colour assignment --------------------------------------------------------------------------------------
# ------------------------------------------ xyz components   
scaler = MinMaxScaler();

alldata = []
for r in [1,2,4,5,6,7]:
    alldata = np.append(np.append(np.append(alldata, (locals()["pca_x"+str(r)]).reshape(-1,1)),(locals()["pca_y"+str(r)]).reshape(-1,1)),(locals()["pca_z"+str(r)]).reshape(-1,1))

scaler.fit(alldata.reshape(-1,1))

for w in [1,2,4,5,6,7]:
    locals()["pca_x"+str(w)] = scaler.transform((locals()["pca_x"+str(w)]).reshape(-1,1));
    locals()["pca_y"+str(w)] = scaler.transform((locals()["pca_y"+str(w)]).reshape(-1,1));
    locals()["pca_z"+str(w)] = scaler.transform((locals()["pca_z"+str(w)]).reshape(-1,1));

for q in [1,2,4,5,6,7]:
    locals()["colourmap_x"+str(q)] = cm.YlOrRd(locals()["pca_x"+str(q)]);
    locals()["colourmap_y"+str(q)] = cm.YlOrRd(locals()["pca_y"+str(q)]);
    locals()["colourmap_z"+str(q)] = cm.YlOrRd(locals()["pca_z"+str(q)]);

for p in [1,2,4,5,6,7]:
    for d in range((locals()["verts"+str(p)]).shape[0]):
        for rgb in range(3):
            (locals()["colours_x"+str(p)])[d,rgb] = (locals()["colourmap_x"+str(p)])[d,0,rgb];
            (locals()["colours_y"+str(p)])[d,rgb] = (locals()["colourmap_y"+str(p)])[d,0,rgb];
            (locals()["colours_z"+str(p)])[d,rgb] = (locals()["colourmap_z"+str(p)])[d,0,rgb];

# save all the colour assigments
for file in [1,2,4,5,6,7]:
    np.save('C:\MPhys\\Data\\Intra Patient\\Stomach_Interpolated\\PCA\\colours_x{0}.npy'.format(file), locals()["colours_x"+str(file)])
    np.save('C:\MPhys\\Data\\Intra Patient\\Stomach_Interpolated\\PCA\\colours_y{0}.npy'.format(file), locals()["colours_y"+str(file)])
    np.save('C:\MPhys\\Data\\Intra Patient\\Stomach_Interpolated\\PCA\\colours_z{0}.npy'.format(file), locals()["colours_z"+str(file)])
    
# ------------------------------------------- magnitudes
 # Import the PCA results for all patients
pca_result_cubeMag1 = np.load('C:\MPhys\\Data\\Intra Patient\\Stomach_Interpolated\\PCA\\pcaMagPanc01.npy')
pca_result_cubeMag2 = np.load('C:\MPhys\\Data\\Intra Patient\\Stomach_Interpolated\\PCA\\pcaMagStomach02.npy')
pca_result_cubeMag4 = np.load('C:\MPhys\\Data\\Intra Patient\\Stomach_Interpolated\\PCA\\pcaMagStomach04.npy')
pca_result_cubeMag5 = np.load('C:\MPhys\\Data\\Intra Patient\\Stomach_Interpolated\\PCA\\pcaMagStomach05.npy')
pca_result_cubeMag6 = np.load('C:\MPhys\\Data\\Intra Patient\\Stomach_Interpolated\\PCA\\pcaMagStomach06.npy')
pca_result_cubeMag7 = np.load('C:\MPhys\\Data\\Intra Patient\\Stomach_Interpolated\\PCA\\pcaMagStomach07.npy')
    
# flip the pca data in the same manner in order to ensure the data for each point remains consistent
pca_result_cubeMagLR1 = np.fliplr(pca_result_cubeMag1)
pca_result_cubeMagLR2 = np.fliplr(pca_result_cubeMag2)
pca_result_cubeMagLR4 = np.fliplr(pca_result_cubeMag4)
pca_result_cubeMagLR5 = np.fliplr(pca_result_cubeMag5)
pca_result_cubeMagLR6 = np.fliplr(pca_result_cubeMag6)
pca_result_cubeMagLR7 = np.fliplr(pca_result_cubeMag7) 

# creating arrays
for yu in [1,2,4,5,6,7]:
    locals()["pcaMag"+str(yu)] = np.ndarray(shape = ((locals()["verts"+str(yu)]).shape[0]))
    locals()["coloursMag"+str(yu)] = np.ndarray(shape = ((locals()["verts"+str(yu)]).shape[0],3))

# filling arrays
for er in [1,2,4,5,6,7]:
    for i in range((locals()["verts"+str(er)]).shape[0]):
        (locals()["pcaMag"+str(er)])[i] = (locals()["pca_result_cubeMagLR"+str(er)])[(locals()["verts_round"+str(er)])[i,0],(locals()["verts_round"+str(er)])[i,1],(locals()["verts_round"+str(er)])[i,2],0];
  
# appending all arrays to a list
alldata2 = []
for r2 in [1,2,4,5,6,7]:
    alldata2 = np.append(alldata2, (locals()["pcaMag"+str(r2)]).reshape(-1,1))
# scale the data
scaler.fit(alldata2.reshape(-1,1))

# transform the data
for w2 in [1,2,4,5,6,7]:
    locals()["pcaMag"+str(w2)] = scaler.transform((locals()["pcaMag"+str(w2)]).reshape(-1,1));


for q2 in [1,2,4,5,6,7]:
    locals()["colourMap"+str(q2)] = cm.YlOrRd(locals()["pcaMag"+str(q2)]);


for p2 in [1,2,4,5,6,7]:
    for d2 in range((locals()["verts"+str(p2)]).shape[0]):
        for rgb2 in range(3):
            (locals()["coloursMag"+str(p2)])[d2,rgb2] = (locals()["colourMap"+str(p2)])[d2,0,rgb2];

# save np arrays
for file2 in [1,2,4,5,6,7]:
    np.save('C:\MPhys\\Data\\Intra Patient\\Stomach_Interpolated\\PCA\\coloursMag{0}.npy'.format(file2), locals()["coloursMag"+str(file2)])