# Cube data sift
"""
Created on Tue Nov 27 11:14:47 2018

@author: Eleanor
"""

import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import nibabel as nib


def reVectorize(dataVec):
    out = np.zeros((np.shape(dataVec)[0], np.shape(dataVec)[1], np.shape(dataVec)[2],3))
    for i in range(np.shape(dataVec)[0]):
        for j in range(np.shape(dataVec)[1]):
            for k in range(np.shape(dataVec)[2]):
                out[i][j][k][0] = dataVec[i][j][k][0][0]
                out[i][j][k][1] = dataVec[i][j][k][0][1]
                out[i][j][k][2] = dataVec[i][j][k][0][2]
    return out

#import .py file containing PCA info
img = nib.load('C:\MPhys\\Nifti_Images\\cropped\\104\\averageVecs.nii')
niiData = img.get_fdata()
data = reVectorize(niiData)
#normalise data in the range 0-1, to ensure RGB values work
#make it specific to x,y,z
for q in range(3):
    min = np.amin(data[:,:,:,q])
    data[:,:,:,q] = data[:,:,:,q] + abs(min)


for h in range(3):
    max = np.amax(data[:,:,:,h])
    data[:,:,:,h] = np.divide(data[:,:,:,h],max)
  

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
    ax.index = volume.shape[0]//2  #integer division of array length (in x) 
    ax.imshow(np.rot90(volume[ax.index],1,(1,2)))
    fig.canvas.mpl_connect('key_press_event',process_key)
    
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
    
multi_slice_viewer(data)
    
        


            

            


