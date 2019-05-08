# VRML_colour_reassignment.py
#
# This sceript is designed to allow the user to swap colour patches on stomachs
# that have been analysed by t-SNE. This is in an attempt to allocate the same
# colour to the same regions of each stomach. Note: this is a manual process
# just made easier by this script
#
"""
Created on Fri May  3 12:52:33 2019

@author: Edward Henderson
"""
import numpy as np;

def colourSwap(x):
    blue = [0.235294, 0.266667, 0.847059];
    green = [0.062745, 0.549020, 0.121569];
    red = [0.678431, 0.074510, 0.074510];
    yellow = [1.000000, 0.964706, 0.027451];
    purple = [0.545098,0.074510,0.749020];
    # rewire desired swaps here
    if (x[0] == blue[0]):
        return blue;
    if (x[0] == green[0]):
        return yellow;
    if (x[0] == red[0]):
        return red;
    if (x[0] == yellow[0]):
        return purple;
    if (x[0] == purple[0]):
        return green;
    
vrml = open('C:\MPhys\\Visualisation\\TSNE\\Stomach02\\stomach_shell_clustered5_interpolated_thick_allFlipped_final.wrl','r');
vrmlContents = vrml.readlines();
# find the colour data here
for i in range(len(vrmlContents)):
    if (vrmlContents[i] == '	color[\n'):
        colourStart = i;
        break;

for j in range(colourStart+1,len(vrmlContents)):
    if (vrmlContents[j] == ']\n'):
        colourEnd = j;
        break;
# read old colours here and swap
oldColours = np.ndarray(shape = (colourEnd-colourStart-1,3));
newColours = np.ndarray(shape = (colourEnd-colourStart-1,3));
colourNum = 0;
for k in range(colourStart+1,colourEnd):
    oldColours[colourNum] = vrmlContents[k].split();
    newColours[colourNum] = colourSwap(oldColours[colourNum]);
    colourNum += 1;
# write new file here
vrmlNew = open('C:\MPhys\\Visualisation\\TSNE\\Stomach02\\stomach_shell_clustered5_interpolated_thick_allFlipped_final_recoloured.wrl','w');
for Ed in range(colourStart+1):
    vrmlNew.write(vrmlContents[Ed]);
for Edward in range(newColours.shape[0]):
    for Henderson in range(newColours.shape[1]):
        vrmlNew.write(str("{:.6f}".format(newColours[Edward][Henderson])) + "  ");
    vrmlNew.write("\n");
for Gavin in range(colourEnd, len(vrmlContents)):
    vrmlNew.write(vrmlContents[Gavin]);

vrml.close();
vrmlNew.close();