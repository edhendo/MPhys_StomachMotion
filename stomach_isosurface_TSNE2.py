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
from matplotlib import cm
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.ndimage.morphology import binary_fill_holes

###############################################################################
def swap(x,y):
    temp = x;
    x = y;
    y = temp;
    return x, y;

def cart3sph(x,y,z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r

def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:        
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)              

def multi_slice_viewer(volume):
    remove_keymap_conflicts({'j','k'})
    fig,ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[0] //2 
    ax.imshow(volume[ax.index],aspect=1.0,cmap = 'nipy_spectral')
    fig.canvas.mpl_connect('key_press_event',process_key)

def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
    fig.canvas.draw()

def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index -1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])
    

def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index +1) % volume.shape[0] 
    ax.images[0].set_array(volume[ax.index]) 
###############################################################################
tStart = time.time()
data1 =""
np.set_printoptions(precision=4, suppress=True)
#------------------------------------------------------------------------------
# First extract all required warp vectors from the respective nifti images
counter = 0
for i in range(1,11):
    locals()["img"+str(i)] = nib.load('C:\MPhys\\Nifti_Images\\Stomach_Interpolated\\Stomach07\\warp{0}.nii'.format(i+2)) # plus two for the panc deformations
    locals()['hdr'+str(i)] = locals()['img'+str(i)].header
    locals()['data'+str(i)] = np.flipud(np.array(locals()['img'+str(i)].get_fdata())); # flip data here!! from correct output orientataion
    counter = counter + 1
    print("extracted warp vectors from DVF " + str(counter) + " out of 9")
    if counter == 10:
        print("Warning: included refScan")

#------------------------------------------------------------------------------     
# Read in the delineation nifti files using nibabel
stomach = nib.load('C:\MPhys\\Nifti_Images\\Stomach_Interpolated\\Stomach07\\stomachMask.nii');
stomachHdr = stomach.header;
stomachData = stomach.get_fdata();

# numpy array conversion and flip to match the data and patient
stom = np.flipud(np.array(stomachData));


# Use marching cubes to obtain the surface mesh of the stomach/stomach PRV delineations
# input 3d volume - masking data form WM
vertsInner, facesInner, normalsInner, valuesInner = measure.marching_cubes_lewiner(stom, 90)
verts, faces, normals, values = measure.marching_cubes_lewiner(stom, 50)
vertsOuter, facesOuter, normalsOuter, valuesOuter = measure.marching_cubes_lewiner(stom, 10)
# create new array for faces - needs to have 4 components (-1 as fourth)
facesnew = np.ndarray(shape = (faces.shape[0],4))
for i in range(faces.shape[0]):
    for j in range(3):
        facesnew[i][j] = int(faces[i][j])
    facesnew[i][3] = int(-1)     

# round vertex numbers to nearest int
verts_round = (np.around(verts)).astype(int);
vertsInner_round = (np.around(vertsInner)).astype(int);
vertsOuter_round = (np.around(vertsOuter)).astype(int);

# Now form boundary cubes from the inner and outer verts lists
thiccShell = np.zeros(stom.shape);
shellInner = np.zeros(stom.shape);
for j in range(vertsOuter_round.shape[0]):
    thiccShell[vertsOuter_round[j][0],vertsOuter_round[j][1],vertsOuter_round[j][2]] = 1;
thiccShell = binary_fill_holes(thiccShell);

for i in range(vertsInner_round.shape[0]):
    shellInner[vertsInner_round[i][0],vertsInner_round[i][1],vertsInner_round[i][2]] = 1;
shellInner = binary_fill_holes(shellInner);

thiccShell = thiccShell ^ shellInner;

for i in range(vertsInner_round.shape[0]):
    thiccShell[vertsInner_round[i][0],vertsInner_round[i][1],vertsInner_round[i][2]] = 1;

multi_slice_viewer(thiccShell);
#--------------------------------------------------------------------------------------------------------------------------------
# Display resulting triangular mesh
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

# Fancy indexing: `verts[faces]` to generate a collection of triangles
mesh = Poly3DCollection(vertsInner[facesInner])
mesh.set_edgecolor('k')
ax.add_collection3d(mesh)

#set axis labels
ax.set_xlabel("x-axis: L-R")
ax.set_ylabel("y-axis: A-P")
ax.set_zlabel("z-axis: C-C")

#plot axis limits based on mesh dimensions
ax.set_xlim(vertsInner[:,0].min() - 2, vertsInner[:,0].max() + 2)
ax.set_ylim(vertsInner[:,1].min() - 2, vertsInner[:,1].max() + 2)  
ax.set_zlim(vertsInner[:,2].min() - 2, vertsInner[:,2].max() + 2)  

plt.tight_layout()
plt.show()

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

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

# Fancy indexing: `verts[faces]` to generate a collection of triangles
mesh = Poly3DCollection(vertsOuter[facesOuter])
mesh.set_edgecolor('k')
ax.add_collection3d(mesh)

#set axis labels
ax.set_xlabel("x-axis: L-R")
ax.set_ylabel("y-axis: A-P")
ax.set_zlabel("z-axis: C-C")

#plot axis limits based on mesh dimensions
ax.set_xlim(vertsOuter[:,0].min() - 2, vertsOuter[:,0].max() + 2)
ax.set_ylim(vertsOuter[:,1].min() - 2, vertsOuter[:,1].max() + 2)  
ax.set_zlim(vertsOuter[:,2].min() - 2, vertsOuter[:,2].max() + 2)  

plt.tight_layout()
plt.show()
#------------------------------------------------------------------------------           
# Now strip out the stomach data alone and arrange data for tsne here
useThickShell = True;

if (useThickShell):
    # Extract the DVF data corresponding to the thickened shell
    pixels = 0;
    for x in range(thiccShell.shape[0]):
        for y in range(thiccShell.shape[1]):
            for z in range(thiccShell.shape[2]):
                if (thiccShell[x,y,z] == 1):
                    pixels += 1;
    extracted_DVF_Data = np.ndarray((pixels,6*10));
    pixelNum = 0;
    for x in range(thiccShell.shape[0]):
        for y in range(thiccShell.shape[1]):
            for z in range(thiccShell.shape[2]):
                if (thiccShell[x,y,z] == 1):
                    # value within shell
                    eleIndex = 0;
                    for DVFnum in range(1,11):
                        # run over all DVFs
                        az, el, r = cart3sph(locals()['data'+str(DVFnum)][x,y,z,0,0],locals()['data'+str(DVFnum)][x,y,z,0,1],locals()['data'+str(DVFnum)][x,y,z,0,2]);               
                        for ijk in range(3):
                            extracted_DVF_Data[pixelNum,eleIndex] = locals()['data'+str(DVFnum)][x,y,z,0,ijk]
                            eleIndex += 1;
                        extracted_DVF_Data[pixelNum,eleIndex] = r;                     # Magnitude
                        eleIndex += 1;
                        extracted_DVF_Data[pixelNum,eleIndex] = az;                    # Azimuthal
                        eleIndex += 1;
                        extracted_DVF_Data[pixelNum,eleIndex] = el;                    # Elevation
                        eleIndex += 1;
                    pixelNum += 1;

else:
    # Extract the DVF data from the shell alone
    extracted_DVF_Data = np.zeros((verts.shape[0],6*10))   
    for a in range(verts.shape[0]):
        eleIndex = 0;
        for DVFnum in range(1,11):
            # run over all DVFs
            az, el, r = cart3sph(locals()['data'+str(DVFnum)][verts_round[a,0],verts_round[a,1],verts_round[a,2],0,0],locals()['data'+str(DVFnum)][verts_round[a,0],verts_round[a,1],verts_round[a,2],0,1],locals()['data'+str(DVFnum)][verts_round[a,0],verts_round[a,1],verts_round[a,2],0,2]);               
            for ijk in range(3):
                extracted_DVF_Data[a,eleIndex] = locals()['data'+str(DVFnum)][verts_round[a,0],verts_round[a,1],verts_round[a,2],0,ijk]
                eleIndex += 1;
            extracted_DVF_Data[a,eleIndex] = r;                     # Magnitude
            eleIndex += 1;
            extracted_DVF_Data[a,eleIndex] = az;                    # Azimuthal
            eleIndex += 1;
            extracted_DVF_Data[a,eleIndex] = el;                    # Elevation
            eleIndex += 1;
            #extracted_DVF_Data[a,eleIndex] = verts_round[a,0];      # X position
            #eleIndex += 1;
            #extracted_DVF_Data[a,eleIndex] = verts_round[a,1];      # Y position
            #eleIndex += 1;
            #extracted_DVF_Data[a,eleIndex] = verts_round[a,2];      # Z position
            #eleIndex += 1;
        
#------------------------------------------------------------------------------
# Now perform t-SNE analysis

tTSNE = time.time()
tsneResult = TSNE(n_components=2, n_iter=5000, learning_rate=200, verbose=1).fit_transform(extracted_DVF_Data);
print("t-SNE completed in:" + str(np.round(time.time()-tTSNE)) + " seconds")   

# Now reassemble the data cube to align with the stomach model
voxelNum = 0
tsne_result_cube = np.zeros((thiccShell.shape[0],thiccShell.shape[1],thiccShell.shape[2],2));
for a in range(thiccShell.shape[0]):
    for b in range(thiccShell.shape[1]):
        for c in range(thiccShell.shape[2]):
            if (thiccShell[a,b,c] == 1):
                tsne_result_cube[a][b][c][0] = tsneResult[voxelNum,0];
                tsne_result_cube[a][b][c][1] = tsneResult[voxelNum,1];
                voxelNum += 1; 

tsneSurfaceValues = np.ndarray(shape = (verts.shape[0],2));
# find the t-SNE values that correspond with mesh vertices
# then put the t-SNE values that match the rounded vertex values into an array
for i in range(verts.shape[0]):
    tsneSurfaceValues[i,0] = tsne_result_cube[verts_round[i,0],verts_round[i,1],verts_round[i,2],0];
    tsneSurfaceValues[i,1] = tsne_result_cube[verts_round[i,0],verts_round[i,1],verts_round[i,2],1];

plt.figure()
plt.scatter(tsneResult[:,0],tsneResult[:,1],marker='o',s=10)
plt.xlabel("t-SNE Component 1", fontsize = "18")
plt.ylabel("t-SNE Component 2", fontsize = "18")       
plt.show();

plt.figure()
plt.scatter(tsneSurfaceValues[:,0],tsneSurfaceValues[:,1],marker='o',s=10)
plt.xlabel("t-SNE Component 1", fontsize = "18")
plt.ylabel("t-SNE Component 2", fontsize = "18")       
plt.show();

if (useThickShell):
    tsneResult = tsneSurfaceValues;
#------------------------------------------------------------------------------
# Now perform k-means clustering

clusters = 2
kmeans2 = KMeans(n_clusters=clusters, random_state=10).fit(tsneResult);

# The silhouette_score gives the average value for all the samples.
# This gives a perspective into the density and separation of the formed clusters
silhouette_avg = silhouette_score(tsneResult, kmeans2.labels_)
print("For n_clusters =", clusters,
      "The average silhouette_score is :", silhouette_avg)

for i in range(clusters):
    locals()['cluster'+str(i)] = [];
    
for j in range(tsneResult.shape[0]):
    for k in range(clusters):
        if kmeans2.labels_[j] == k:
            locals()['cluster'+str(k)].append(tsneResult[j]);

for l in range(clusters):
    locals()['cluster'+str(l)] = np.array(locals()['cluster'+str(l)]);
'''
plt.figure()
plt.scatter(cluster0[:,0],cluster0[:,1],marker='o',s=10,color='r')
plt.scatter(cluster1[:,0],cluster1[:,1],marker='o',s=10,color='k')
plt.xlabel("t-SNE Component 1", fontsize = "18")
plt.ylabel("t-SNE Component 2", fontsize = "18")       
plt.show();
'''
clusters = 3
kmeans3 = KMeans(n_clusters=clusters, random_state=10).fit(tsneResult);

# The silhouette_score gives the average value for all the samples.
# This gives a perspective into the density and separation of the formed clusters
silhouette_avg = silhouette_score(tsneResult, kmeans3.labels_)
print("For n_clusters =", clusters,
      "The average silhouette_score is :", silhouette_avg)

for i in range(clusters):
    locals()['cluster'+str(i)] = [];
    
for j in range(tsneResult.shape[0]):
    for k in range(clusters):
        if kmeans3.labels_[j] == k:
            locals()['cluster'+str(k)].append(tsneResult[j]);

for l in range(clusters):
    locals()['cluster'+str(l)] = np.array(locals()['cluster'+str(l)]);
'''
plt.figure()
plt.scatter(cluster0[:,0],cluster0[:,1],marker='o',s=10,color='r')
plt.scatter(cluster1[:,0],cluster1[:,1],marker='o',s=10,color='k')
plt.scatter(cluster2[:,0],cluster2[:,1],marker='o',s=10,color='b')
plt.xlabel("t-SNE Component 1", fontsize = "18")
plt.ylabel("t-SNE Component 2", fontsize = "18")       
plt.show();
'''
clusters = 4
kmeans4 = KMeans(n_clusters=clusters, random_state=10).fit(tsneResult);

# The silhouette_score gives the average value for all the samples.
# This gives a perspective into the density and separation of the formed clusters
silhouette_avg = silhouette_score(tsneResult, kmeans4.labels_)
print("For n_clusters =", clusters,
      "The average silhouette_score is :", silhouette_avg)

for i in range(clusters):
    locals()['cluster'+str(i)] = [];
    
for j in range(tsneResult.shape[0]):
    for k in range(clusters):
        if kmeans4.labels_[j] == k:
            locals()['cluster'+str(k)].append(tsneResult[j]);

for l in range(clusters):
    locals()['cluster'+str(l)] = np.array(locals()['cluster'+str(l)]);
'''
plt.figure()
plt.scatter(cluster0[:,0],cluster0[:,1],marker='o',s=10,color='r')
plt.scatter(cluster1[:,0],cluster1[:,1],marker='o',s=10,color='k')
plt.scatter(cluster2[:,0],cluster2[:,1],marker='o',s=10,color='b')
plt.scatter(cluster3[:,0],cluster3[:,1],marker='o',s=10,color='y')
plt.xlabel("t-SNE Component 1", fontsize = "18")
plt.ylabel("t-SNE Component 2", fontsize = "18")       
plt.show();
'''
clusters = 5
kmeans5 = KMeans(n_clusters=clusters, random_state=1000).fit(tsneResult);

# The silhouette_score gives the average value for all the samples.
# This gives a perspective into the density and separation of the formed clusters
silhouette_avg = silhouette_score(tsneResult, kmeans5.labels_)
print("For n_clusters =", clusters,
      "The average silhouette_score is :", silhouette_avg)

for i in range(clusters):
    locals()['cluster'+str(i)] = [];
    
for j in range(tsneResult.shape[0]):
    for k in range(clusters):
        if kmeans5.labels_[j] == k:
            locals()['cluster'+str(k)].append(tsneResult[j]);

for l in range(clusters):
    locals()['cluster'+str(l)] = np.array(locals()['cluster'+str(l)]);

plt.figure()
plt.scatter(cluster0[:,0],cluster0[:,1],marker='o',s=10,color='r')
plt.scatter(cluster1[:,0],cluster1[:,1],marker='o',s=10,color='k')
plt.scatter(cluster2[:,0],cluster2[:,1],marker='o',s=10,color='b')
plt.scatter(cluster3[:,0],cluster3[:,1],marker='o',s=10,color='y')
plt.scatter(cluster4[:,0],cluster4[:,1],marker='o',s=10,color='g')
plt.xlabel("t-SNE Component 1", fontsize = "18")
plt.ylabel("t-SNE Component 2", fontsize = "18")       
plt.show();

clusters = 6
kmeans6 = KMeans(n_clusters=clusters, random_state=100).fit(tsneResult);

# The silhouette_score gives the average value for all the samples.
# This gives a perspective into the density and separation of the formed clusters
silhouette_avg = silhouette_score(tsneResult, kmeans6.labels_)
print("For n_clusters =", clusters,
      "The average silhouette_score is :", silhouette_avg)

for i in range(clusters):
    locals()['cluster'+str(i)] = [];
    
for j in range(tsneResult.shape[0]):
    for k in range(clusters):
        if kmeans6.labels_[j] == k:
            locals()['cluster'+str(k)].append(tsneResult[j]);

for l in range(clusters):
    locals()['cluster'+str(l)] = np.array(locals()['cluster'+str(l)]);
'''
plt.figure()
plt.scatter(cluster0[:,0],cluster0[:,1],marker='o',s=10,color='r')
plt.scatter(cluster1[:,0],cluster1[:,1],marker='o',s=10,color='k')
plt.scatter(cluster2[:,0],cluster2[:,1],marker='o',s=10,color='b')
plt.scatter(cluster3[:,0],cluster3[:,1],marker='o',s=10,color='y')
plt.scatter(cluster4[:,0],cluster4[:,1],marker='o',s=10,color='g')
plt.scatter(cluster5[:,0],cluster5[:,1],marker='o',s=10,color='c')
plt.xlabel("t-SNE Component 1", fontsize = "18")
plt.ylabel("t-SNE Component 2", fontsize = "18")       
plt.show();
'''
clusters = 7
kmeans7 = KMeans(n_clusters=clusters, random_state=100).fit(tsneResult);

# The silhouette_score gives the average value for all the samples.
# This gives a perspective into the density and separation of the formed clusters
silhouette_avg = silhouette_score(tsneResult, kmeans7.labels_)
print("For n_clusters =", clusters,
      "The average silhouette_score is :", silhouette_avg)

for i in range(clusters):
    locals()['cluster'+str(i)] = [];
    
for j in range(tsneResult.shape[0]):
    for k in range(clusters):
        if kmeans7.labels_[j] == k:
            locals()['cluster'+str(k)].append(tsneResult[j]);

for l in range(clusters):
    locals()['cluster'+str(l)] = np.array(locals()['cluster'+str(l)]);
'''
plt.figure()
plt.scatter(cluster0[:,0],cluster0[:,1],marker='o',s=10,color='r')
plt.scatter(cluster1[:,0],cluster1[:,1],marker='o',s=10,color='k')
plt.scatter(cluster2[:,0],cluster2[:,1],marker='o',s=10,color='b')
plt.scatter(cluster3[:,0],cluster3[:,1],marker='o',s=10,color='y')
plt.scatter(cluster4[:,0],cluster4[:,1],marker='o',s=10,color='g')
plt.scatter(cluster5[:,0],cluster5[:,1],marker='o',s=10,color='c')
plt.scatter(cluster6[:,0],cluster6[:,1],marker='o',s=10,color='m')
plt.xlabel("t-SNE Component 1", fontsize = "18")
plt.ylabel("t-SNE Component 2", fontsize = "18")       
plt.show();
'''
#------------------------------------------------------------------------------
# Now assign colour values to each of the t-SNE clusters
tsne_vertex_colours5 = np.ndarray((verts.shape[0],3));
#tsne_vertex_colours6 = np.ndarray((verts.shape[0],3));
#tsne_vertex_colours7 = np.ndarray((verts.shape[0],3));

for i in range(verts.shape[0]):
    if kmeans5.labels_[i] == 0:
        tsne_vertex_colours5[i] = [60/255,68/255,216/255];
    if kmeans5.labels_[i] == 1:
        tsne_vertex_colours5[i] = [16/255,140/255,31/255];
    if kmeans5.labels_[i] == 2:
        tsne_vertex_colours5[i] = [173/255,19/255,19/255];
    if kmeans5.labels_[i] == 3:
        tsne_vertex_colours5[i] = [255/255,246/255,7/255];
    if kmeans5.labels_[i] == 4:
        tsne_vertex_colours5[i] = [139/255,19/255,191/255];
    #for rgb in range(3):
        #tsne_vertex_colours5[i,rgb] = cm.Set1(kmeans5.labels_[i])[rgb];
        #tsne_vertex_colours6[i,rgb] = cm.Set1(kmeans6.labels_[i])[rgb];
        #tsne_vertex_colours7[i,rgb] = cm.Set1(kmeans7.labels_[i])[rgb];

#------------------------------------------------------------------------------
######################## Perform VRML file write here #########################

wrlFile5 = open('C:\MPhys\\Visualisation\\TSNE\\Stomach07\\stomach_shell_clustered5_interpolated_thick_allFlipped_final.wrl','w');
wrlFile5.write('#VRML V2.0 utf8\nWorldInfo {title "stomach_shell_clustered5_interpolated_thick_allFlipped_final"}\n  Shape {\n   appearance Appearance { material Material{ transparency  0.0 } }\n   geometry IndexedFaceSet {\n    coord DEF surf1 Coordinate{\n	point [\n');  

for i in range(verts.shape[0]):
    for j in range(verts.shape[1]):
        wrlFile5.write(str("{:.6f}".format(verts[i][j])) + "  ");
    wrlFile5.write("\n");              

wrlFile5.write("]}\n	color Color {\n	color[\n");
           
for i in range(tsne_vertex_colours5.shape[0]):
    for j in range(tsne_vertex_colours5.shape[1]):
        wrlFile5.write(str("{:.6f}".format(tsne_vertex_colours5[i][j])) + "  ");
    wrlFile5.write("\n");
    
wrlFile5.write("]\n	}\n	colorPerVertex TRUE	\n	coordIndex [\n");
    
for i in range(faces.shape[0]):
    faces[i,1], faces[i,2] = swap(faces[i,1], faces[i,2]);
    for j in range(3):
        wrlFile5.write(str(int(faces[i][j])) + "  ");
    wrlFile5.write(str(int(-1))+ "\n");              

wrlFile5.write("	]\n	}\n}");
wrlFile5.close();

'''
wrlFile6 = open('C:\MPhys\\Visualisation\\TSNE\\Stomach07\\stomach_shell_clustered6_interpolated_thick.wrl','w');
wrlFile6.write('#VRML V2.0 utf8\nWorldInfo {title "stomach_shell_clustered5_interpolated_thick"}\n  Shape {\n   appearance Appearance { material Material{ transparency  0.0 } }\n   geometry IndexedFaceSet {\n    coord DEF surf1 Coordinate{\n	point [\n');  

for i in range(verts.shape[0]):
    for j in range(verts.shape[1]):
        wrlFile6.write(str("{:.6f}".format(verts[i][j])) + "  ");
    wrlFile6.write("\n");              

wrlFile6.write("]}\n	color Color {\n	color[\n");
           
for i in range(tsne_vertex_colours6.shape[0]):
    for j in range(tsne_vertex_colours6.shape[1]):
        wrlFile6.write(str("{:.6f}".format(tsne_vertex_colours6[i][j])) + "  ");
    wrlFile6.write("\n");
    
wrlFile6.write("]\n	}\n	colorPerVertex TRUE	\n	coordIndex [\n");
    
for i in range(faces.shape[0]):
    for j in range(3):
        wrlFile6.write(str(int(faces[i][j])) + "  ");
    wrlFile6.write(str(int(-1))+ "\n");              

wrlFile6.write("	]\n	}\n}");
wrlFile6.close();


wrlFile7 = open('C:\MPhys\\Visualisation\\TSNE\\Stomach07\\just_shell_clustered7.wrl','w');
#wrlFile7 = open('D:\data\\Pancreas\\MPhys\\TSNE results\\stomachTSNE.wrl','w');
wrlFile7.write('#VRML V2.0 utf8\nWorldInfo {title "just_shell_clustered7"}\n  Shape {\n   appearance Appearance { material Material{ transparency  0.0 } }\n   geometry IndexedFaceSet {\n    coord DEF surf1 Coordinate{\n	point [\n');  

for i in range(verts.shape[0]):
    for j in range(verts.shape[1]):
        wrlFile7.write(str("{:.6f}".format(verts[i][j])) + "  ");
    wrlFile7.write("\n");              

wrlFile7.write("]}\n	color Color {\n	color[\n");
           
for i in range(tsne_vertex_colours7.shape[0]):
    for j in range(tsne_vertex_colours7.shape[1]):
        wrlFile7.write(str("{:.6f}".format(tsne_vertex_colours7[i][j])) + "  ");
    wrlFile7.write("\n");
    
wrlFile7.write("]\n	}\n	colorPerVertex TRUE	\n	coordIndex [\n");
    
for i in range(faces.shape[0]):
    faces[i,1], faces[i,2] = swap(faces[i,1], faces[i,2]);
    for j in range(3):
        wrlFile7.write(str(int(faces[i][j])) + "  ");
    wrlFile7.write(str(int(-1))+ "\n");              

wrlFile7.write("	]\n	}\n}");
wrlFile7.close();
'''
print("Program completed in: " + str(np.round(time.time()-tStart)) + " seconds")





















