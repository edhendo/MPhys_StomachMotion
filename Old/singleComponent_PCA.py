# singleComponent_PCA.py
# formally whole_image_PCA.py
"""
Created on Tue Dec  4 14:37:24 2018

@author: Edward Henderson
"""
import time
import numpy as np
import nibabel as nib
import math
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

tStart = time.time()
data1 =""
np.set_printoptions(precision=4, suppress=True)

def cart3sph(x,y,z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r

def magnitude(x,y,z):
    return math.sqrt((x**2 + y**2 + z**2))

# Define which scan is the reference scan, aka the maximum exhale scan
refScanNum = 12

# First extract all required warp vectors from the respective nifti images
counter = 0
for i in range(1,11):
    if i != refScanNum: #Not needed right now since the reference scan is set to -1*averagewarp
        locals()["img"+str(i)] = nib.load('C:\MPhys\\Nifti_Images\\cropped\\104\\warp{0}.nii'.format(i))
        locals()['hdr'+str(i)] = locals()['img'+str(i)].header
        locals()['data'+str(i)] = locals()['img'+str(i)].get_fdata()
        counter = counter + 1
        print("extracted warp vectors from DVF " + str(counter) + " out of 9")
        if counter == 10:
            print("Warning: included refScan")

# Perform PCA over the whole image with only magnitude r
# voxel| DVF1 | DVF2  | ...  | DVF10
#   1  | r11  |  r12  | r1.  | r110
#   2  | r21  |  r22  | r2.  | r210
#  ... | r..1 |  r..2 | r... | r..10
#   M  | rM1  |  rM2  | rM.  | rM10

# M is total number of voxels eg M = 49*89*51 dependent on cropping
# exclude the reference scan (only use 9 DVFs)
# patient 104 has 49x89x51

# first construct the big matrix
m = 0
#azData = np.zeros((data1.shape[0]*data1.shape[1]*data1.shape[2],9))
#elData = np.zeros((data1.shape[0]*data1.shape[1]*data1.shape[2],9))
rData = np.zeros((data1.shape[0]*data1.shape[1]*data1.shape[2],counter))
xData = np.zeros((data1.shape[0]*data1.shape[1]*data1.shape[2],counter))
yData = np.zeros((data1.shape[0]*data1.shape[1]*data1.shape[2],counter))
zData = np.zeros((data1.shape[0]*data1.shape[1]*data1.shape[2],counter))
# fill the matrix
t1 = time.time()
for x in range(data1.shape[0]):
    for y in range(data1.shape[1]):
        for z in range(data1.shape[2]):
            DVFindex = 0
            xyzIndex = 0
            for DVFnum in range(1,11):
                if DVFnum != refScanNum:
                    az, el, r = cart3sph(locals()['data'+str(DVFnum)][x][y][z][0][0],locals()['data'+str(DVFnum)][x][y][z][0][1],locals()['data'+str(DVFnum)][x][y][z][0][2])
                    #azData[m][DVFindex] = az
                    #elData[m][DVFindex] = el
                    rData[m][DVFindex] = r
                    xData[m][DVFindex] = locals()['data'+str(DVFnum)][x][y][z][0][0]
                    yData[m][DVFindex] = locals()['data'+str(DVFnum)][x][y][z][0][1]
                    zData[m][DVFindex] = locals()['data'+str(DVFnum)][x][y][z][0][2]
                    DVFindex = DVFindex + 1
            m = m + 1
print("Filled huge matrix in:" + str(np.round(time.time()-t1)) + " seconds")

# PCA

t2 = time.time()
Rpca = PCA(n_components=9)
#AZpca = PCA(n_components=9)
#ELpca = PCA(n_components=9)
Xpca = PCA(n_components=9)
Ypca = PCA(n_components=9)
Zpca = PCA(n_components=9)


Rpca_result = Rpca.fit_transform(rData)
#AZpca_result = AZpca.fit_transform(azData)
#ELpca_result = ELpca.fit_transform(elData)
Xpca_result = Xpca.fit_transform(xData)
Ypca_result = Ypca.fit_transform(yData)
Zpca_result = Zpca.fit_transform(zData)

print("PCA completed in:" + str(np.round(time.time()-t2)) + " seconds")
#print("Explained variation in az per principal component: {}".format(AZpca.explained_variance_ratio_))
#print("Explained variation in el per principal component: {}".format(ELpca.explained_variance_ratio_))
print("Explained variation in r per principal component: {}".format(Rpca.explained_variance_ratio_))
print("Explained variation in x per principal component: {}".format(Xpca.explained_variance_ratio_))
#print("Explained variation in y per principal component: {}".format(Ypca.explained_variance_ratio_))
#print("Explained variation in z per principal component: {}".format(Zpca.explained_variance_ratio_))

# Now read the principle components from the PCA back into a data cube for slice by slice visualisation

t3 = time.time()
min_max_scaler = MinMaxScaler()
for PCAcompIndex in range(9):
    locals()['Rcomponent' + str(PCAcompIndex + 1)] = np.zeros((data1.shape[0] * data1.shape[1] * data1.shape[2],1))
    locals()['Xcomponent' + str(PCAcompIndex + 1)] = np.zeros((data1.shape[0] * data1.shape[1] * data1.shape[2],1))
    for voxelNum in range(data1.shape[0] * data1.shape[1] * data1.shape[2]):
        locals()['Rcomponent' + str(PCAcompIndex + 1)][voxelNum] = Rpca_result[voxelNum][PCAcompIndex]
        locals()['Xcomponent' + str(PCAcompIndex + 1)][voxelNum] = Rpca_result[voxelNum][PCAcompIndex]
    locals()['scaledRComponent' + str(PCAcompIndex + 1)] = min_max_scaler.fit_transform((locals()['Rcomponent' + str(PCAcompIndex + 1)]).reshape(-1,1))
    locals()['scaledXComponent' + str(PCAcompIndex + 1)] = min_max_scaler.fit_transform((locals()['Xcomponent' + str(PCAcompIndex + 1)]).reshape(-1,1))

voxelNum = 0
Rpca_result_cube = np.zeros((data1.shape[0],data1.shape[1],data1.shape[2],9))
Xpca_result_cube = np.zeros((data1.shape[0],data1.shape[1],data1.shape[2],9))
for x3 in range(data1.shape[0]):
    for y3 in range(data1.shape[1]):
        for z3 in range(data1.shape[2]):
            for PCAcompIndex2 in range(9):
                Rpca_result_cube[x3][y3][z3][PCAcompIndex2] = locals()['Rcomponent' + str(PCAcompIndex2 + 1)][voxelNum]
                Xpca_result_cube[x3][y3][z3][PCAcompIndex2] = locals()['Xcomponent' + str(PCAcompIndex2 + 1)][voxelNum]
            voxelNum = voxelNum + 1

print("Data reshaped in: " + str(np.round(time.time()-t3)) + " seconds")

# end
print("Program completed in: " + str(np.round(time.time()-tStart)) + " seconds")


# Now save the resultant PCA data as .npy arrays

np.save('C:\MPhys\\Data\\singleCompPCA\\Rpca104_result_cube.npy', Rpca_result_cube)
np.save('C:\MPhys\\Data\\singleCompPCA\\Xpca104_result_cube.npy', Xpca_result_cube)
# accessed through np.load(path)