# plotly visualisation
"""
Created on Thu Mar 21 15:25:28 2019

@author: Eleanor
"""
import plotly
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
#mag_pca_result_cube = np.load('C:\MPhys\\Data\\PCA results\\Panc01StomachCropMagnitudePCAcube.npy');
mag_pca_result_cube = np.load('C:\MPhys\\Data\\Intra Patient\\Stomach\\PCA\\pcaMagPanc01.npy');

toggle = True; # set to True for using pca on magnitudes rather than magnitude of pca comps

# Read in the delineation nifti files using nibabel
#stomach = nib.load('C:\MPhys\\stomach.nii');
stomach = nib.load('C:\MPhys\\Data\\Intra Patient\\Stomach\\Panc01_NR\\stomach.nii')
stomachHdr = stomach.header;
stomachData = stomach.get_fdata();

# numpy array conversion
#stom = np.rot90(np.rot90(np.array(stomachData),2,(0,2)),1,(1,2));
stom = np.array(stomachData);

# Use marching cubes to obtain the surface mesh of the stomach/stomach PRV delineations
# input 3d volume - masking data form WM
verts, faces, normals, values = measure.marching_cubes_lewiner(stom, 50) # note to self:check masking boudaries in lua code - CHECKED eh
x,y,z = zip(*verts)
#i,j,k = zip(*faces)


#find the PCA vector values that correspond with mesh vertices
#put the PCA values that match the rounded vertex values into an array
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

#ff.create_trisurf
data = go.Mesh3d(x,y,z, faces, colormap='RdBu', plot_edges=None, vertexcolor = colours)

title="Panc01 - PCA Magnitudes"

fig3 = go.Figure(data=data)
fig3['layout'].update(dict(title=title,
                           width=1000,
                           height=1000,
                           scene=dict(
                                      aspectratio=dict(x=1, y=1, z=0.4),
                                      camera=dict(eye=dict(x=1.25, y=1.25, z= 1.25)
                                     )
                           )
                     ))

plotly.offline.plot(fig3, filename = 'Panc01 - PCA Magnitudes')