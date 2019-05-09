# pca_slice_viewer.py
"""
Created on Thu Dec  6 11:32:07 2018

@author: Edward Henderson
"""

import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

pca = np.load('C:\MPhys\\Data\\PCA results\\niftyregPanc01StomachCropPCAshell.npy');
#pca = np.load('C:\\MPhys\\Data\\Intra Patient\\Pancreas\\PCA\\pcaShellPanc01.npy')
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
    ax.imshow(volume[ax.index],aspect=1.0,cmap = 'nipy_spectral')
    plt.xlabel("L-R direction");
    plt.ylabel("C-C direction");
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

rotated = np.rot90(np.rot90(pca,1,(1,2)),2,(0,1))
multi_slice_viewer(rotated[:,:,:,0,0])
