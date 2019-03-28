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
from skimage import measure
import math
from sklearn.preprocessing import MinMaxScaler
from matplotlib import cm
from MulticoreTSNE import MulticoreTSNE as multiTSNE
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_samples, silhouette_score

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
    locals()["img"+str(i)] = nib.load('C:\MPhys\\Nifti_Images\\Stomach07\\warp{0}.nii'.format(i+2)) # plus two for the panc deformations
    #locals()["img"+str(i)] = nib.load('D:\data\\Pancreas\\MPhys\\Nifti_Images\\Stomach\\Stomach07\\warp{0}.nii'.format(i+2)) # plus two for the panc deformations
    locals()['hdr'+str(i)] = locals()['img'+str(i)].header
    locals()['data'+str(i)] = locals()['img'+str(i)].get_fdata()
    counter = counter + 1
    print("extracted warp vectors from DVF " + str(counter) + " out of 9")
    if counter == 10:
        print("Warning: included refScan")

#------------------------------------------------------------------------------     
# Read in the delineation nifti files using nibabel
stomach = nib.load('C:\MPhys\\Nifti_Images\\Stomach07\\stomachMask.nii');
# stomach = nib.load('C:\MPhys\\Data\\Intra Patient\\Pancreas\\niftyregStomach07StomachCrop\\stomach.nii')
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
# Now strip out the stoamch data alone and arrange data for pca here
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
tsneResult = TSNE(n_components=2, n_iter=1000, learning_rate=200).fit_transform(extracted_DVF_Data);
print("t-SNE completed in:" + str(np.round(time.time()-tTSNE)) + " seconds")   
plt.figure()
plt.scatter(tsneResult[:,0],tsneResult[:,1],marker='o',s=10)
plt.xlabel("t-SNE Component 1", fontsize = "18")
plt.ylabel("t-SNE Component 2", fontsize = "18")       
plt.show();
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

plt.figure()
plt.scatter(cluster0[:,0],cluster0[:,1],marker='o',s=10,color='r')
plt.scatter(cluster1[:,0],cluster1[:,1],marker='o',s=10,color='k')
plt.xlabel("t-SNE Component 1", fontsize = "18")
plt.ylabel("t-SNE Component 2", fontsize = "18")       
plt.show();

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

plt.figure()
plt.scatter(cluster0[:,0],cluster0[:,1],marker='o',s=10,color='r')
plt.scatter(cluster1[:,0],cluster1[:,1],marker='o',s=10,color='k')
plt.scatter(cluster2[:,0],cluster2[:,1],marker='o',s=10,color='b')
plt.xlabel("t-SNE Component 1", fontsize = "18")
plt.ylabel("t-SNE Component 2", fontsize = "18")       
plt.show();

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

plt.figure()
plt.scatter(cluster0[:,0],cluster0[:,1],marker='o',s=10,color='r')
plt.scatter(cluster1[:,0],cluster1[:,1],marker='o',s=10,color='k')
plt.scatter(cluster2[:,0],cluster2[:,1],marker='o',s=10,color='b')
plt.scatter(cluster3[:,0],cluster3[:,1],marker='o',s=10,color='y')
plt.xlabel("t-SNE Component 1", fontsize = "18")
plt.ylabel("t-SNE Component 2", fontsize = "18")       
plt.show();

clusters = 5
kmeans5 = KMeans(n_clusters=clusters, random_state=10).fit(tsneResult);

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
kmeans6 = KMeans(n_clusters=clusters, random_state=10).fit(tsneResult);

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

clusters = 7
kmeans7 = KMeans(n_clusters=clusters, random_state=10).fit(tsneResult);

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
#------------------------------------------------------------------------------
# Now assign colour values to each of the t-SNE clusters
tsne_vertex_colours5 = np.ndarray((verts.shape[0],3));
tsne_vertex_colours6 = np.ndarray((verts.shape[0],3));
tsne_vertex_colours7 = np.ndarray((verts.shape[0],3));

for i in range(verts.shape[0]):
    for rgb in range(3):
        tsne_vertex_colours5[i,rgb] = cm.Dark2(kmeans5.labels_[i])[rgb];
        tsne_vertex_colours6[i,rgb] = cm.Dark2(kmeans6.labels_[i])[rgb];
        tsne_vertex_colours7[i,rgb] = cm.Dark2(kmeans7.labels_[i])[rgb];

#------------------------------------------------------------------------------
######################## Perform VRML file write here #########################

wrlFile5 = open('C:\MPhys\\Visualisation\\TSNE\\Stomach07\\just_shell_clustered5.wrl','w');
#wrlFile5 = open('D:\data\\Pancreas\\MPhys\\TSNE results\\stomachTSNE.wrl','w');
wrlFile5.write('#VRML V2.0 utf8\nWorldInfo {title "just_shell_clustered5"}\n  Shape {\n   appearance Appearance { material Material{ transparency  0.1 } }\n   geometry IndexedFaceSet {\n    coord DEF surf1 Coordinate{\n	point [\n');  

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
    for j in range(3):
        wrlFile5.write(str(int(faces[i][j])) + "  ");
    wrlFile5.write(str(int(-1))+ "\n");              

wrlFile5.write("	]\n	}\n}");
wrlFile5.close();

wrlFile6 = open('C:\MPhys\\Visualisation\\TSNE\\Stomach07\\just_shell_clustered6.wrl','w');
#wrlFile6 = open('D:\data\\Pancreas\\MPhys\\TSNE results\\stomachTSNE.wrl','w');
wrlFile6.write('#VRML V2.0 utf8\nWorldInfo {title "just_shell_clustered6"}\n  Shape {\n   appearance Appearance { material Material{ transparency  0.1 } }\n   geometry IndexedFaceSet {\n    coord DEF surf1 Coordinate{\n	point [\n');  

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
wrlFile7.write('#VRML V2.0 utf8\nWorldInfo {title "just_shell_clustered7"}\n  Shape {\n   appearance Appearance { material Material{ transparency  0.1 } }\n   geometry IndexedFaceSet {\n    coord DEF surf1 Coordinate{\n	point [\n');  

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
    for j in range(3):
        wrlFile7.write(str(int(faces[i][j])) + "  ");
    wrlFile7.write(str(int(-1))+ "\n");              

wrlFile7.write("	]\n	}\n}");
wrlFile7.close();

print("Program completed in: " + str(np.round(time.time()-tStart)) + " seconds")





















