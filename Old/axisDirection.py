# axisDirection.py
"""
Created on Tue Dec 11 15:51:56 2018

@author: Edward Henderson
"""

import time
import numpy as np
import nibabel as nib
import math
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt


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

testLungs = nib.load('C:\MPhys\\Nifti_Images\\testLungs.nii').get_fdata()

# first construct the big matrix
m = 0
data = np.zeros((testLungs.shape[0]*testLungs.shape[1]*testLungs.shape[2]))
# fill the matrix
t1 = time.time()
for x in range(testLungs.shape[0]):
    for y in range(testLungs.shape[1]):
        for z in range(testLungs.shape[2]):
            data[m] = testLungs[x][y][z]
            m = m + 1
print("Filled huge matrix in:" + str(np.round(time.time()-t1)) + " seconds")

# Now read the principle components from the PCA back into a data cube for slice by slice visualisation

t3 = time.time()
voxelNum = 0
testLungs_cube = np.zeros((testLungs.shape[0],testLungs.shape[1],testLungs.shape[2]))

for x3 in range(testLungs.shape[0]):
    for y3 in range(testLungs.shape[1]):
        for z3 in range(testLungs.shape[2]):
            testLungs_cube[x3][y3][z3]= data[voxelNum]
            voxelNum = voxelNum + 1

print("Data reshaped in: " + str(np.round(time.time()-t3)) + " seconds")

# end
print("Program completed in: " + str(np.round(time.time()-tStart)) + " seconds")

#plot image slice using imshow()
#use transpose to switch to horizontal slices
#assign keys j and k to enable us to move forward/backwar through the slices
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
    ax.volume = volume #unsure of meaning
    ax.index = volume.shape[0] //2  #integer division of array length (in x) 
    ax.imshow(volume[ax.index],aspect=1.0,cmap = "OrRd")
    fig.canvas.mpl_connect('key_press_event',process_key)
#Colormap r is not recognized. Possible values are: Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r, Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys, Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, Paired, Paired_r, Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, Set1, Set1_r, Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r, Vega10, Vega10_r, Vega20, Vega20_r, Vega20b, Vega20b_r, Vega20c, Vega20c_r, Wistia, Wistia_r, YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r, afmhot, afmhot_r, autumn, autumn_r, binary, binary_r, bone, bone_r, brg, brg_r, bwr, bwr_r, cool, cool_r, coolwarm, coolwarm_r, copper, copper_r, cubehelix, cubehelix_r, flag, flag_r, gist_earth, gist_earth_r, gist_gray, gist_gray_r, gist_heat, gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r, gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r, inferno, inferno_r, jet, jet_r, magma, magma_r, nipy_spectral, nipy_spectral_r, ocean, ocean_r, pink, pink_r, plasma, plasma_r, prism, prism_r, rainbow, rainbow_r, seismic, seismic_r, spectral, spectral_r, spring, spring_r, summer, summer_r, tab10, tab10_r, tab20, tab20_r, tab20b, tab20b_r, tab20c, tab20c_r, terrain, terrain_r, viridis, viridis_r, winter, winter_r
def process_key(event):
    fig = event.canvas.figure #plots figure on a canvas?
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
    fig.canvas.draw()

def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index -1) % volume.shape[0]  #unsure about meaning?? 
    ax.images[0].set_array(volume[ax.index])
    

def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index +1) % volume.shape[0] 
    ax.images[0].set_array(volume[ax.index]) 
    
multi_slice_viewer(np.rot90((testLungs_cube[:,:,:]),3,(1,2)))












