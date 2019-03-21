# stomach_isosurface2.py
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
import math
from sklearn.preprocessing import MinMaxScaler
from matplotlib import cm

def magnitude(x,y,z):
    return math.sqrt((x**2 + y**2 + z**2))

tStart = time.time();
np.set_printoptions(precision=4, suppress=True);
<<<<<<< HEAD
pca_result_cube = np.load('C:\MPhys\\Data\\PCA results\\niftyregPanc01StomachCropPCAcube.npy');
#pca_result_cube = np.load('C:\MPhys\\Data\\Intra Patient\\Stomach\\PCA\\pcaPanc01_NR.npy')
mag_pca_result_cube = np.load('C:\MPhys\\Data\\PCA results\\Panc01StomachCropMagnitudePCAcube.npy');
#mag_pca_result_cube = np.load('C:\MPhys\\Data\\Intra Patient\\Stomach\\PCA\\pcaMagPanc01_NR.npy');
=======
#pca_result_cube = np.load('C:\MPhys\\Data\\PCA results\\niftyregPanc01StomachCropPCAcube.npy');
pca_result_cube = np.load('C:\MPhys\\Data\\Intra Patient\\Stomach\\PCA\\pcaPanc01_NR.npy')
#mag_pca_result_cube = np.load('C:\MPhys\\Data\\PCA results\\Panc01StomachCropMagnitudePCAcube.npy');
<<<<<<< HEAD
mag_pca_result_cube = np.load('C:\MPhys\\Data\\Intra Patient\\Stomach\\PCA\\pcaMagPanc01_NR.npy');
=======
mag_pca_result_cube = np.load('C:\MPhys\\Data\\Intra Patient\\Stomach\\PCA\\pcaMagStomach02.npy');
>>>>>>> 093bf2b35f2f6bf35ec9a1653ad9a26dd323346f
>>>>>>> f6be48addf01f0acf68f8b0b2b325eca426cca99

toggle = True; # set to True for using pca on magnitudes rather than magnitude of pca comps

# Read in the delineation nifti files using nibabel
<<<<<<< HEAD
stomach = nib.load('C:\MPhys\\stomach.nii');
#stomach = nib.load('C:\MPhys\\Data\\Intra Patient\\Stomach\\Panc01_NR\\stomach.nii')
=======
#stomach = nib.load('C:\MPhys\\stomach.nii');
<<<<<<< HEAD
stomach = nib.load('C:\MPhys\\Data\\Intra Patient\\Stomach\\Panc01_NR\\stomach.nii')
=======
stomach = nib.load('C:\MPhys\\Data\\Intra Patient\\Stomach\\Stomach02\\stomachMask.nii')
>>>>>>> 093bf2b35f2f6bf35ec9a1653ad9a26dd323346f
>>>>>>> f6be48addf01f0acf68f8b0b2b325eca426cca99
stomachHdr = stomach.header;
stomachData = stomach.get_fdata();
'''
stomach_PRV = nib.load('C:\MPhys\\Data\\Intra Patient\\Pancreas\\niftyregPanc01StomachCrop\\stomach_PRVMask.nii');
stomach_PRVHdr = stomach_PRV.header;
stomach_PRVData = stomach_PRV.get_fdata();
'''
# numpy array conversion
#stom = np.rot90(np.rot90(np.array(stomachData),2,(0,2)),1,(1,2));
stom = np.array(stomachData);
#stomPRV= np.array(stomach_PRVData);

# Use marching cubes to obtain the surface mesh of the stomach/stomach PRV delineations
# input 3d volume - masking data form WM
verts, faces, normals, values = measure.marching_cubes_lewiner(stom, 50) #note to self:check masking boudaries in lua code 

#------------------ save vertex and face coordinates into txt files --------------------------------------------------------------
<<<<<<< HEAD
#np.savetxt('C:\MPhys\\Data\\Intra Patient\\Pancreas\\3D Vis\\stomachVerts01.txt',verts, fmt = '%0.6f')
#np.savetxt('C:\MPhys\\Visualisation\\stomachVerts01.txt',verts, fmt = '%0.6f');

=======
>>>>>>> 093bf2b35f2f6bf35ec9a1653ad9a26dd323346f
# create new array for faces - needs to have 4 components (-1 as fourth)
facesnew = np.ndarray(shape = (faces.shape[0],4))
for i in range(faces.shape[0]):
    for j in range(3):
        facesnew[i][j] = int(faces[i][j])
    facesnew[i][3] = int(-1)
    
<<<<<<< HEAD
#np.savetxt('C:\MPhys\\Data\\Intra Patient\\Pancreas\\3D Vis\\stomachFaces01.txt',facesnew.astype(int), fmt = '%0.0f')
#np.savetxt('C:\MPhys\\Visualisation\\stomachFaces01.txt',facesnew.astype(int), fmt = '%0.0f')
=======
>>>>>>> 093bf2b35f2f6bf35ec9a1653ad9a26dd323346f
#--------------------------------------------------------------------------------------------------------------------------------

# Display resulting triangular mesh
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

# Fancy indexing: `verts[faces]` to generate a collection of triangles
mesh = Poly3DCollection(verts[faces])
mesh.set_edgecolor('k')
ax.add_collection3d(mesh)

#set axis labels
ax.set_xlabel("x-axis: L-R")
ax.set_ylabel("y-axis: A-P")
ax.set_zlabel("z-axis: C-C")

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
coloursMag = np.ndarray(shape = (verts.shape[0]))
#round vertex numbers to nearest int
verts_round = (np.around(verts)).astype(int)

# fill a linear array of the magnitude of the PCA components
PCAmagnitudes = np.zeros((pca_result_cube.shape[0],pca_result_cube.shape[1],pca_result_cube.shape[2]));

for x in range(pca_result_cube.shape[0]):
    for y in range(pca_result_cube.shape[1]):
        for z in range(pca_result_cube.shape[2]):
            PCAmagnitudes[x,y,z] = magnitude(pca_result_cube[x,y,z,0,0],pca_result_cube[x,y,z,0,1],pca_result_cube[x,y,z,0,2]);

for x1 in range(verts.shape[0]):
    coloursMag[x1] = PCAmagnitudes[verts_round[x1,0],verts_round[x1,1],verts_round[x1,2]];

if toggle:
    for x2 in range(verts.shape[0]):
        coloursMag[x2] = mag_pca_result_cube[verts_round[x2,0],verts_round[x2,1],verts_round[x2,2],0];

scaler = MinMaxScaler();
coloursMag = scaler.fit_transform(coloursMag.reshape(-1,1));

colourmap = cm.rainbow(coloursMag);

for x in range(verts.shape[0]):
    for l in range(3):
        colours[x,l] = colourmap[x,0,l];
                                        
<<<<<<< HEAD
#write into a text file
#np.savetxt('C:\MPhys\\Data\\Intra Patient\\Pancreas\\3D Vis\\stomachColours01.txt', colours, fmt = '%0.6f')
#np.savetxt('C:\MPhys\\Visualisation\\stomachColours01.txt', colours, fmt = '%0.6f')


=======
>>>>>>> 093bf2b35f2f6bf35ec9a1653ad9a26dd323346f
## --> have one vis method showing the magnitude of each point with a single color map
## --> then separate x,y,z components
## --> T-SNE similarly

# Do the file writing here
<<<<<<< HEAD
'''
wrlFile = open('C:\MPhys\\Visualisation\\stomachPCA_mag.wrl','w');
=======
wrlFile = open('C:\MPhys\\Data\\Intra Patient\\Stomach\\3D Vis\\pcaMagPanc01.wrl','w');
#wrlFile = open('C:\MPhys\\Visualisation\\stomachPCA_mag.wrl','w');
>>>>>>> 093bf2b35f2f6bf35ec9a1653ad9a26dd323346f
wrlFile.write('#VRML V2.0 utf8\nWorldInfo {title "stomach-PCA-VRML"}\n  Shape {\n   appearance Appearance { material Material{ transparency  0.1 } }\n   geometry IndexedFaceSet {\n    coord DEF surf1 Coordinate{\n	point [\n');  

for i in range(verts.shape[0]):
    for j in range(verts.shape[1]):
        wrlFile.write(str("{:.6f}".format(verts[i][j])) + "  ");
    wrlFile.write("\n");              

wrlFile.write("]}\n	color Color {\n	color[\n");
           
for i in range(colours.shape[0]):
    for j in range(colours.shape[1]):
        wrlFile.write(str("{:.6f}".format(colours[i][j])) + "  ");
    wrlFile.write("\n");
    
wrlFile.write("]\n	}\n	colorPerVertex TRUE	\n	coordIndex [\n");
    
for i in range(faces.shape[0]):
    for j in range(3):
        wrlFile.write(str(int(faces[i][j])) + "  ");
    wrlFile.write(str(int(-1))+ "\n");              

wrlFile.write("	]\n	}\n}");
wrlFile.close();

print("Program completed in: " + str(np.round(time.time()-tStart)) + " seconds")
'''