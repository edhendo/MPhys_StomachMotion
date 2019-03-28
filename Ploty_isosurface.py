# plotly visualisation
"""
Created on Thu Mar 21 15:25:28 2019

@author: Eleanor
"""
import plotly
plotly.tools.set_credentials_file(username='eleanor1357', api_key='oB200N0KGmaub83XuktV')
plotly.tools.set_config_file(world_readable=False,
                             sharing='private')
import plotly.plotly as py
import plotly.figure_factory as ff
import plotly.graph_objs as go
from skimage import measure
import numpy as np
import time
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler
from matplotlib import cm
 



tStart = time.time();
np.set_printoptions(precision=4, suppress=True);
############################## load data ########################
#mag_pca_result_cube = np.load('C:\MPhys\\Data\\PCA results\\Panc01StomachCropMagnitudePCAcube.npy');
mag_pca_result_cube = np.load('C:\MPhys\\Data\\Intra Patient\\Stomach\\PCA\\pcaMagStomach04.npy');


# Read in the delineation nifti files using nibabel
#stomach = nib.load('C:\MPhys\\stomach.nii');
stomach = nib.load('C:\MPhys\\Data\\Intra Patient\\Stomach\\Stomach04\\stomachMask.nii')
stomachHdr = stomach.header;
stomachData = stomach.get_fdata();

# numpy array conversion
# stom = np.rot90(np.rot90(np.array(stomachData),2,(0,2)),1,(1,2));
stom = np.array(stomachData);

<<<<<<< HEAD
##################### functions ###############################

def tri_indices(faces):
    #simplices is a numpy array defining the simplices of the triangularization
    #returns the lists of indices i, j, k

    return ([triplet[c] for triplet in faces] for c in range(3))
=======
# Use marching cubes to obtain the surface mesh of the stomach/stomach PRV delineations
# input 3d volume - masking data form WM
verts, faces, normals, values = measure.marching_cubes_lewiner(stom, 50) # note to self:check masking boudaries in lua code - CHECKED eh
x,y,z = zip(*verts)
#i,j,k = zip(*faces)
>>>>>>> 0589cd3f380f6b59a837106c15e3391b5cc20311

################ create trisurf plot #########################

# Use marching cubes to obtain the surface mesh of the stomach/stomach PRV delineations
# input 3d volume - masking data form WM
verts, faces, normals, values = measure.marching_cubes_lewiner(stom, 50)
X3,Y3,Z3 = zip(*verts)
X3 = np.array(X3)
Y3 = np.array(Y3)
Z3 = np.array(Z3)
i,j,k = tri_indices(faces)

# find the PCA vector values that correspond with mesh vertices and put the values that match the rounded vertex values into an array
colours = np.ndarray(shape = (verts.shape[0],3))
coloursMag = np.ndarray(shape = (verts.shape[0]))
#round vertex numbers to nearest int
verts_round = (np.around(verts)).astype(int)

for x2 in range(verts.shape[0]):
    coloursMag[x2] = mag_pca_result_cube[verts_round[x2,0],verts_round[x2,1],verts_round[x2,2],0];

scaler = MinMaxScaler();
coloursMag = scaler.fit_transform(coloursMag.reshape(-1,1));

colourmap = cm.YlOrRd(coloursMag);

for X in range(verts.shape[0]):
    for l in range(3):
        colours[X,l] = colourmap[X,0,l];
        
#function to assign colours to vertices
def vertexColour(x,y,z):
    for h in range(X3.size):
        if X3[h] == x:
            Index = h
    return colours(Index)
        
        
data = ff.create_trisurf(x=X3, y=Y3, z=Z3, simplices = faces, color_func = vertexColour, show_colorbar = True)
#data = [go.Mesh3d(x=x,y=y,z=z, i=i,j=j,k=k, vertexcolor = colours, colorscale = 'YlOrRd', colorbar = dict(thickness = 20, x = 2))]

fig3 = go.Figure(data=data)
fig3['layout'].update(dict(title= ' Stomach04 - PCA Magnitudes',
                           width=1000,
                           height=1000,
                           scene=dict(
                                      aspectratio=dict(x=1, y=1, z=0.4),
                                      camera=dict(eye=dict(x=1.25, y=1.25, z= 1.25)
                                     )
                           )
                     ))

plotly.offline.plot(fig3, filename = 'Stomach04 - PCA Magnitudes.html')