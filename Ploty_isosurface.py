# plotly visualisation
"""
Created on Thu Mar 21 15:25:28 2019
@author: Eleanor
"""
import plotly
plotly.tools.set_credentials_file(username='eleanor1357', api_key='oB200N0KGmaub83XuktV')
plotly.tools.set_config_file(world_readable=True,
                             sharing='public')
import plotly.plotly as py
import plotly.graph_objs as go
from skimage import measure
import numpy as np
import time
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler
from matplotlib import cm

tStart = time.time();
np.set_printoptions(precision=4, suppress=True);
#---------- load data and create mesh ----------------------------
#mag_pca_result_cube = np.load('C:\MPhys\\Data\\PCA results\\Panc01StomachCropMagnitudePCAcube.npy');
mag_pca_result_cube = np.load('C:\MPhys\\Data\\Intra Patient\\Stomach\\PCA\\pcaMagStomach07.npy');
#pca_result_cube = np.load('C:\MPhys\\Data\\PCA results\\Stomach04PCAcube.npy');
pca_result_cube = np.load('C:\MPhys\\Data\\Intra Patient\\Stomach\\PCA\\pcaStomach07.npy') 

# Read in the delineation nifti files using nibabel
#stomach = nib.load('C:\MPhys\\stomach.nii');
stomach = nib.load('C:\MPhys\\Data\\Intra Patient\\Stomach\\Stomach07\\stomachMask.nii')
stomachHdr = stomach.header;
stomachData = stomach.get_fdata();

# numpy array conversion
# stom = np.rot90(np.rot90(np.array(stomachData),2,(0,2)),1,(1,2));
stom = np.array(stomachData);

##################### functions ###############################

def tri_indices(faces):
    #simplices is a numpy array defining the simplices of the triangularization
    #returns the lists of indices i, j, k

    return ([triplet[c] for triplet in faces] for c in range(3))

################ create trisurf plot #########################

# Use marching cubes to obtain the surface mesh of the stomach/stomach PRV delineations
# input 3d volume - masking data form WM
verts, faces, normals, values = measure.marching_cubes_lewiner(stom, 50)
x,y,z = zip(*verts)
x = np.array(x)
y = np.array(y)
z = np.array(z)
I,J,K = tri_indices(faces)

#------------------------------------------------ Magnitude graph plotting ---------------------------------------------

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

color_trace = go.Scatter(x=[0 for _ in colours],
                            y=[0 for _ in colours],
                            mode='markers',
                            marker= dict(
                                colorscale= 'YlOrRd',
                                reversescale = True,
                                size=1,
                                colorbar = dict(x=0.9199, y=0.5, len = 0.5),
                                color=colours,
                                showscale=True,
                                )
                            )    

data = [go.Mesh3d(x=x,y=y,z=z, i=I,j=J,k=K, vertexcolor = colours), color_trace]

fig = go.Figure(data=data)
fig['layout'].update(dict(title= 'Stomach07 - PCA Magnitudes',
                            xaxis = dict(type = 'linear', showticklabels = False, showgrid = False,
                                        zeroline = False,
                                        autorange = True),
                            yaxis = dict(type = 'linear', showticklabels = False, showgrid = False,
                                        zeroline = False,
                                        autorange = True),
                            width = 1000,
                            height = 1000,
                            scene = dict(xaxis = dict(type = 'linear', showticklabels = False),
                                        yaxis = dict(type = 'linear', showticklabels = False),
                                        zaxis = dict(type = 'linear', showticklabels = False),
                                        aspectratio=dict(x=1, y=1, z=0.75),
                                        camera=dict(up = dict(x=0,y=0,z=1), eye=dict(x=-1.51, y=1.373, z=0.248), center=dict(x=0,y=0,z=0)
                                        )
                           )
                     ))
py.plot(fig, filename = 'Stomach07 - PCA Magnitudes.html')

#-------------------------------------------------- x,y,z graph plotting ------------------------------------------------------
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
    pca_x[i] = pca_result_cube[verts_round[i,0],verts_round[i,1],verts_round[i,2],0,0];
    pca_y[i] = pca_result_cube[verts_round[i,0],verts_round[i,1],verts_round[i,2],0,1];
    pca_z[i] = pca_result_cube[verts_round[i,0],verts_round[i,1],verts_round[i,2],0,2];

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
        
color_trace2 = go.Scatter(x=[0 for _ in colours], y=[0 for _ in colours], mode='markers',
                            marker= dict(colorscale= 'YlOrRd', reversescale = True, size=1,
                                colorbar = dict(x=0.9199, y=0.5, len = 0.5), color=colours, showscale=True,
                                )
                            ) 

datax = [go.Mesh3d(x=x,y=y,z=z, i=I,j=J,k=K, vertexcolor = colours_x), color_trace2]
datay = [go.Mesh3d(x=x,y=y,z=z, i=I,j=J,k=K, vertexcolor = colours_y), color_trace2]
dataz = [go.Mesh3d(x=x,y=y,z=z, i=I,j=J,k=K, vertexcolor = colours_z), color_trace2]

figx = go.Figure(data = datax)
figy = go.Figure(data = datay)
figz = go.Figure(data = dataz)

figx['layout'].update(dict(title= 'Stomach07 - x component',
                            xaxis = dict(type = 'linear', showticklabels = False, showgrid = False, zeroline = False, autorange = True),
                            yaxis = dict(type = 'linear', showticklabels = False, showgrid = False, zeroline = False, autorange = True),
                            width = 1000,
                            height = 1000,
                            scene = dict(xaxis = dict(type = 'linear', showticklabels = False),
                                        yaxis = dict(type = 'linear', showticklabels = False),
                                        zaxis = dict(type = 'linear', showticklabels = False),
                                        aspectratio=dict(x=1, y=1, z=0.75),
                                        camera=dict(up = dict(x=0,y=0,z=1), eye=dict(x=-1.51, y=1.373, z=0.248), center=dict(x=0,y=0,z=0)
                                        )
                           )
                     ))

figy['layout'].update(dict(title= 'Stomach07 - y component',
                            xaxis = dict(type = 'linear', showticklabels = False, showgrid = False, zeroline = False, autorange = True),
                            yaxis = dict(type = 'linear', showticklabels = False, showgrid = False, zeroline = False, autorange = True),
                            width = 1000,
                            height = 1000,
                            scene = dict(xaxis = dict(type = 'linear', showticklabels = False),
                                        yaxis = dict(type = 'linear', showticklabels = False),
                                        zaxis = dict(type = 'linear', showticklabels = False),
                                        aspectratio=dict(x=1, y=1, z=0.75),
                                        camera=dict(up = dict(x=0,y=0,z=1), eye=dict(x=-1.51, y=1.373, z=0.248), center=dict(x=0,y=0,z=0)
                                        )
                           )
                     ))
                                        
figz['layout'].update(dict(title= 'Stomach07 - z component',
                            xaxis = dict(type = 'linear', showticklabels = False, showgrid = False, zeroline = False, autorange = True),
                            yaxis = dict(type = 'linear', showticklabels = False, showgrid = False, zeroline = False, autorange = True),
                            width = 1000,
                            height = 1000,
                            scene = dict(xaxis = dict(type = 'linear', showticklabels = False),
                                        yaxis = dict(type = 'linear', showticklabels = False),
                                        zaxis = dict(type = 'linear', showticklabels = False),
                                        aspectratio=dict(x=1, y=1, z=0.75),
                                        camera=dict(up = dict(x=0,y=0,z=1), eye=dict(x=-1.51, y=1.373, z=0.248), center=dict(x=0,y=0,z=0)
                                        )
                           )
                     ))

py.plot(figx, filename = 'Stomach07 - x component')
py.plot(figy, filename = 'Stomach07 - y component')
py.plot(figz, filename = 'Stomach07 - z component')
