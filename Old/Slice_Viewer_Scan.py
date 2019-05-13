# Slice_Viewer_Scan.py
"""
Created on Thu Nov 29 11:34:11 2018

@author: Edward Henderson
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

#img = nib.load('C:\MPhys\\Nifti_Images\\lung103scan.nii')
#niiData = img.get_fdata()
'''
norms101 = nib.load('C:\MPhys\\Nifti_Images\\cropped\\101\\averageNorms.nii')
norms102 = nib.load('C:\MPhys\\Nifti_Images\\cropped\\102\\averageNorms.nii')
norms103 = nib.load('C:\MPhys\\Nifti_Images\\cropped\\103\\averageNorms.nii')
norms104 = nib.load('C:\MPhys\\Nifti_Images\\cropped\\104\\averageNorms.nii')
norms101Data = norms101.get_fdata()
norms102Data = norms102.get_fdata()
norms103Data = norms103.get_fdata()
norms104Data = norms104.get_fdata()
testLungs = nib.load('C:\MPhys\\Nifti_Images\\testLungs.nii').get_fdata()
'''
pca104 = np.load('C:\MPhys\\Data\\PCA results\\pcaLung104fixed2.npy')

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
    remove_keymap_conflicts({'j','k','h','l'})
    fig,ax = plt.subplots()
    ax.volume = volume #unsure of meaning
    ax.index = volume.shape[0] //2  #integer division of array length (in x) 
    ax.imshow(volume[ax.index],aspect=1.0)
    fig.canvas.mpl_connect('key_press_event',process_key)
    
def process_key(event):
    fig = event.canvas.figure #plots figure on a canvas?
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
    elif event.key == 'l':
        next_slice_jump(ax)
    elif event.key == 'h':
        previous_slice_jump(ax)
    fig.canvas.draw()

def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index -1) % volume.shape[0]  #unsure about meaning?? 
    ax.images[0].set_array(volume[ax.index])
    

def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index +1) % volume.shape[0] 
    ax.images[0].set_array(volume[ax.index])

def next_slice_jump(ax):
    volume = ax.volume
    ax.index = (ax.index +25) % volume.shape[0] 
    ax.images[0].set_array(volume[ax.index]) 

def previous_slice_jump(ax):
    volume = ax.volume
    ax.index = (ax.index -25) % volume.shape[0] 
    ax.images[0].set_array(volume[ax.index]) 

multi_slice_viewer()
