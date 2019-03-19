# stomach_isosurface_TSNE.py
"""
Created on Thu Mar 14 17:07:37 2019

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
from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE as multiTSNE
#from mpl_toolkits.mplot3d import Axes3D

toggle = True; ### Toggle to pre-load the tsne data cube if already filled

def cart3sph(x,y,z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r

def magnitude(x,y,z):
    return math.sqrt((x**2 + y**2 + z**2))

tStart = time.time()
data1 =""
np.set_printoptions(precision=4, suppress=True)

# Define which scan is the reference scan, aka the maximum exhale scan
#------------------------------------------------------------------------------
# PANC01 has maxExhale at 9
maxExhale = 9
#------------------------------------------------------------------------------
# First extract all required warp vectors from the respective nifti images
counter = 0
for i in range(1,11):
    locals()["img"+str(i)] = nib.load('C:\MPhys\\Nifti_Images\\niftyregPanc01StomachCrop\\warp{0}.nii'.format(i+2)) # plus two for the panc deformations
    locals()['hdr'+str(i)] = locals()['img'+str(i)].header
    locals()['data'+str(i)] = locals()['img'+str(i)].get_fdata()
    counter = counter + 1
    print("extracted warp vectors from DVF " + str(counter) + " out of 9")
    if counter == 10:
        print("Warning: included refScan")

# fill the matrix for t-SNE analysis
tMatFill = time.time()
if (toggle):
    dataMatrix = np.load('C:\MPhys\\Data\\TSNE results\\panc01_StomachCrop_TSNEdataCube.npy');
else:
    dataMatrix = np.zeros((data1.shape[0]*data1.shape[1]*data1.shape[2],9*10))
    m = 0
    xIndex = 0
    yIndex = 0
    zIndex = 0
    
    for x in range(data1.shape[0]):
        for y in range(data1.shape[1]):
            for z in range(data1.shape[2]):
                eleIndex = 0
                for DVFnum in range(1,11):
                    az, el, r = cart3sph(locals()['data'+str(DVFnum)][x][y][z][0][0],locals()['data'+str(DVFnum)][x][y][z][0][1],locals()['data'+str(DVFnum)][x][y][z][0][2])
                    for j in range(3):
                        dataMatrix[m][eleIndex] = locals()['data'+str(DVFnum)][x][y][z][0][j]
                        eleIndex += 1
                    dataMatrix[m][eleIndex] = az
                    eleIndex += 1
                    dataMatrix[m][eleIndex] = el
                    eleIndex += 1
                    dataMatrix[m][eleIndex] = r
                    eleIndex += 1
                    dataMatrix[m][eleIndex] = x    # also give it the original voxel poisitions?!
                    eleIndex += 1
                    dataMatrix[m][eleIndex] = y
                    eleIndex += 1
                    dataMatrix[m][eleIndex] = z
                    eleIndex += 1
                m = m + 1
    np.save('C:\MPhys\\Data\\TSNE results\\panc01_StomachCrop_TSNEdataCube.npy', dataMatrix)
    
print("Filled huge matrix in: " + str(np.round(time.time()-tMatFill)) + " seconds")

# perform voxel-by-voxel t-SNE analysis
tTSNE = time.time()
tsneResult = TSNE(n_components=1, n_iter=1500, learning_rate=175).fit_transform(dataMatrix);
print("t-SNE completed in:" + str(np.round(time.time()-tTSNE)) + " seconds")
np.save('C:\MPhys\\Data\\TSNE results\\panc01_StomachCrop_TSNEresult.npy', tsneResult)
'''
plt.figure()
plt.scatter(tsneResult[:,0],tsneResult[:,1],marker='.',s=0.25)
plt.xlabel("t-SNE Component 1", fontsize = "20")
plt.ylabel("t-SNE Component 2", fontsize = "20")
'''
###############################################################################
# --> Now reassemble the data cube to align with the stomach model
voxelNum = 0
tsne_result_cube = np.zeros((data1.shape[0],data1.shape[1],data1.shape[2]));

for a in range(data1.shape[0]):
    for b in range(data1.shape[1]):
        for c in range(data1.shape[2]):
            tsne_result_cube[a][b][c] = tsneResult[voxelNum];
            voxelNum += 1;

###############################################################################
# Read in the delineation nifti files using nibabel
stomach = nib.load('C:\MPhys\\stomach.nii');
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

#----------------- assign t-SNE colour values ---------------------------------
tsneValues = np.ndarray(shape = (verts.shape[0]));
tsne_colours = np.ndarray(shape = (verts.shape[0],3));
# round vertex numbers to nearest int
verts_round = (np.around(verts)).astype(int)
# find the t-SNE values that correspond with mesh vertices
# then put the t-SNE values that match the rounded vertex values into an array
for i in range(verts.shape[0]):
    tsneValues[i] = tsne_result_cube[verts_round[i,0],verts_round[i,1],verts_round[i,2]];

scaler = MinMaxScaler();
tsneValues = scaler.fit_transform(tsneValues.reshape(-1,1));
# assign colours using colour map here
tsne_colourmap = cm.terrain(tsneValues);

for j in range(verts.shape[0]):
    for rgb in range(3):
        tsne_colours[j,rgb] = tsne_colourmap[j,0,rgb];
#------------------------------------------------------------------------------
######################## Perform VRML file write here #########################
# --> Firstly the x component
wrlFile = open('C:\MPhys\\Visualisation\\stomachTSNE.wrl','w');
wrlFile.write('#VRML V2.0 utf8\nWorldInfo {title "stomachTSNE"}\n  Shape {\n   appearance Appearance { material Material{ transparency  0.1 } }\n   geometry IndexedFaceSet {\n    coord DEF surf1 Coordinate{\n	point [\n');  

for i in range(verts.shape[0]):
    for j in range(verts.shape[1]):
        wrlFile.write(str("{:.6f}".format(verts[i][j])) + "  ");
    wrlFile.write("\n");              

wrlFile.write("]}\n	color Color {\n	color[\n");
           
for i in range(tsne_colours.shape[0]):
    for j in range(tsne_colours.shape[1]):
        wrlFile.write(str("{:.6f}".format(tsne_colours[i][j])) + "  ");
    wrlFile.write("\n");
    
wrlFile.write("]\n	}\n	colorPerVertex TRUE	\n	coordIndex [\n");
    
for i in range(faces.shape[0]):
    for j in range(3):
        wrlFile.write(str(int(faces[i][j])) + "  ");
    wrlFile.write(str(int(-1))+ "\n");              

wrlFile.write("	]\n	}\n}");
wrlFile.close();

print("Program completed in: " + str(np.round(time.time()-tStart)) + " seconds")

