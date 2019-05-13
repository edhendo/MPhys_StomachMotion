# stomach_isosurfaceXYZ
"""
Created on Thu Mar 14 12:53:21 2019

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
#pca_result_cube = np.load('C:\MPhys\\Data\\PCA results\\Stomach07PCAcube.npy');
pca_result_cube = np.load('C:\MPhys\\Data\\Intra Patient\\Stomach_Interpolated\\PCA\\pcaStomach07.npy')

# Read in the delineation nifti files using nibabel
#stomach = nib.load('C:\MPhys\\Nifti_Images\\Stomach07\\stomachMask.nii');
stomach = nib.load('C:\MPhys\\Data\\Intra Patient\\Stomach_Interpolated\\Stomach07\\stomachMask.nii')
stomachHdr = stomach.header;
stomachData = stomach.get_fdata();

# numpy array conversion
#stom = np.rot90(np.rot90(np.array(stomachData),2,(0,2)),1,(1,2));
stom = np.array(stomachData);
stomLR = np.fliplr(stom)
    
# flip the pca data in the same manner in order to ensure the data for each point remains consistent
pca_result_cubeLR = np.fliplr(pca_result_cube)


# Use marching cubes to obtain the surface mesh of the stomach/stomach PRV delineations
# input 3d volume - masking data form WM
verts, faces, normals, values = measure.marching_cubes_lewiner(stomLR, 50)

# create new array for faces - needs to have 4 components (-1 as fourth)
facesnew = np.ndarray(shape = (faces.shape[0],4))
for i in range(faces.shape[0]):
    for j in range(3):
        facesnew[i][j] = int(faces[i][j])
    facesnew[i][3] = int(-1)

#--------------------------------------------------------------------------------------------------------------------------------
'''
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
#plt.show()
'''
#----------------- assign PCA colour values --------------------------------------------------------------------------------------
#find the PCA vector values that correspond with mesh vertices
#put the PCA values that match the rounded vertex values into an array
# separate x, y and z components here
pca_x = np.ndarray(shape = (verts.shape[0]))
pca_y = np.ndarray(shape = (verts.shape[0]))
pca_z = np.ndarray(shape = (verts.shape[0]))
colours_x = np.ndarray(shape = (verts.shape[0],3))
colours_y = np.ndarray(shape = (verts.shape[0],3))
colours_z = np.ndarray(shape = (verts.shape[0],3))
#round vertex numbers to nearest int
verts_round = (np.around(verts)).astype(int)

for i in range(verts.shape[0]):
    pca_x[i] = pca_result_cubeLR[verts_round[i,0],verts_round[i,1],verts_round[i,2],0,0];
    pca_y[i] = pca_result_cubeLR[verts_round[i,0],verts_round[i,1],verts_round[i,2],0,1];
    pca_z[i] = pca_result_cubeLR[verts_round[i,0],verts_round[i,1],verts_round[i,2],0,2];

scaler = MinMaxScaler();

scaler.fit(np.append(np.append(pca_x.reshape(-1,1),pca_y.reshape(-1,1)),pca_z.reshape(-1,1)).reshape(-1,1))

pca_x = scaler.transform(pca_x.reshape(-1,1));
pca_y = scaler.transform(pca_y.reshape(-1,1));
pca_z = scaler.transform(pca_z.reshape(-1,1));

colourmap_x = cm.YlOrRd(pca_x);
colourmap_y = cm.YlOrRd(pca_y);
colourmap_z = cm.YlOrRd(pca_z);

for j in range(verts.shape[0]):
    for rgb in range(3):
        colours_x[j,rgb] = colourmap_x[j,0,rgb];
        colours_y[j,rgb] = colourmap_y[j,0,rgb];
        colours_z[j,rgb] = colourmap_z[j,0,rgb];

# Do the file writing here
# --> Firstly the x component
wrlFileX = open('C:\MPhys\\Data\\Intra Patient\\Stomach_Interpolated\\3D Vis\\pcaStomach07_x.wrl','w');
#wrlFileX = open('C:\MPhys\\Visualisation\\Stomach07\\stomachPCA_x_LR.wrl','w');
wrlFileX.write('#VRML V2.0 utf8\nWorldInfo {title "stomach-PCA-VRML-x_component"}\n  Shape {\n   appearance Appearance { material Material{ transparency  0.1 } }\n   geometry IndexedFaceSet {\n    coord DEF surf1 Coordinate{\n	point [\n');  

for i in range(verts.shape[0]):
    for j in range(verts.shape[1]):
        wrlFileX.write(str("{:.6f}".format(verts[i][j])) + "  ");
    wrlFileX.write("\n");              

wrlFileX.write("]}\n	color Color {\n	color[\n");
           
for i in range(colours_x.shape[0]):
    for j in range(colours_x.shape[1]):
        wrlFileX.write(str("{:.6f}".format(colours_x[i][j])) + "  ");
    wrlFileX.write("\n");
    
wrlFileX.write("]\n	}\n	colorPerVertex TRUE	\n	coordIndex [\n");
    
for i in range(faces.shape[0]):
    for j in range(3):
        wrlFileX.write(str(int(faces[i][j])) + "  ");
    wrlFileX.write(str(int(-1))+ "\n");              

wrlFileX.write("	]\n	}\n}");
wrlFileX.close();

# --> now y
wrlFileY = open('C:\MPhys\\Data\\Intra Patient\\Stomach_Interpolated\\3D Vis\\pcaStomach07_y.wrl','w');
#wrlFileY = open('C:\MPhys\\Visualisation\\Stomach07\\stomachPCA_y_AP.wrl','w');
wrlFileY.write('#VRML V2.0 utf8\nWorldInfo {title "stomach-PCA-VRML-y_component"}\n  Shape {\n   appearance Appearance { material Material{ transparency  0.1 } }\n   geometry IndexedFaceSet {\n    coord DEF surf1 Coordinate{\n	point [\n');  

for i in range(verts.shape[0]):
    for j in range(verts.shape[1]):
        wrlFileY.write(str("{:.6f}".format(verts[i][j])) + "  ");
    wrlFileY.write("\n");              

wrlFileY.write("]}\n	color Color {\n	color[\n");
           
for i in range(colours_y.shape[0]):
    for j in range(colours_y.shape[1]):
        wrlFileY.write(str("{:.6f}".format(colours_y[i][j])) + "  ");
    wrlFileY.write("\n");
    
wrlFileY.write("]\n	}\n	colorPerVertex TRUE	\n	coordIndex [\n");
    
for i in range(faces.shape[0]):
    for j in range(3):
        wrlFileY.write(str(int(faces[i][j])) + "  ");
    wrlFileY.write(str(int(-1))+ "\n");              

wrlFileY.write("	]\n	}\n}");
wrlFileY.close();

# --> now z
wrlFileZ = open('C:\MPhys\\Data\\Intra Patient\\Stomach_Interpolated\\3D Vis\\pcaStomach07_z.wrl','w');
#wrlFileZ = open('C:\MPhys\\Visualisation\\Stomach07\\stomachPCA_z_CC.wrl','w');
wrlFileZ.write('#VRML V2.0 utf8\nWorldInfo {title "stomach-PCA-VRML-z_component"}\n  Shape {\n   appearance Appearance { material Material{ transparency  0.1 } }\n   geometry IndexedFaceSet {\n    coord DEF surf1 Coordinate{\n	point [\n');  

for i in range(verts.shape[0]):
    for j in range(verts.shape[1]):
        wrlFileZ.write(str("{:.6f}".format(verts[i][j])) + "  ");
    wrlFileZ.write("\n");              

wrlFileZ.write("]}\n	color Color {\n	color[\n");
           
for i in range(colours_z.shape[0]):
    for j in range(colours_z.shape[1]):
        wrlFileZ.write(str("{:.6f}".format(colours_z[i][j])) + "  ");
    wrlFileZ.write("\n");
    
wrlFileZ.write("]\n	}\n	colorPerVertex TRUE	\n	coordIndex [\n");
    
for i in range(faces.shape[0]):
    for j in range(3):
        wrlFileZ.write(str(int(faces[i][j])) + "  ");
    wrlFileZ.write(str(int(-1))+ "\n");              

wrlFileZ.write("	]\n	}\n}");
wrlFileZ.close();


print("Program completed in: " + str(np.round(time.time()-tStart)) + " seconds")
