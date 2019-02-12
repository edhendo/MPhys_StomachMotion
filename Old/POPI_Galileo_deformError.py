# POPI_Galileo_deformError.py
"""
Created on Thu Nov 15 12:20:17 2018

@author: Edward Henderson
"""

import numpy as np
import nibabel as nib
import csv

np.set_printoptions(precision=2, suppress=True)

for i in range(1,11):
    if i != 7:
        locals()["img"+str(i)] = nib.load('C:\MPhys\\Nifti_Images\\POPI\\warp{0}.nii'.format(i))
        locals()['hdr'+str(i)] = locals()['img'+str(i)].header
        locals()['data'+str(i)] = locals()['img'+str(i)].get_fdata()

for i in range(10):
    with open("C:\MPhys\Data\POPI-model\landmarks\{0}0-landmarks.txt".format(i)) as landmarksFile:
        reader = csv.reader(landmarksFile, delimiter = '\t')
        locals()['landmarks'+str(i)] = list(reader)
    del locals()['landmarks'+str(i)][0]
    del locals()['landmarks'+str(i)][len(locals()['landmarks'+str(i)])-1]

tempMin = 1000
tempMax = 0
for i in np.linspace(0,9,10):
    for j in np.linspace(0,39,40):
        for k in [0,1,2]:
            if float(locals()['landmarks'+str(int(i))][int(j)][k])<tempMin:
                tempMin = float(locals()['landmarks'+str(int(i))][int(j)][k])
            if float(locals()['landmarks'+str(int(i))][int(j)][k])>tempMax:
                tempMax = float(locals()['landmarks'+str(int(i))][int(j)][k])

print(tempMax)
print(tempMin)