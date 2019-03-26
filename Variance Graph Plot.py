# Variance Graph Plot
"""
Created on Tue Mar 26 16:30:54 2019

@author: Eleanor
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# import all PCA data for magnitudes
panc01 = np.load('C:\MPhys\\Data\\Intra Patient\\Stomach\\PCA\\EVR_panc01.npy')
stomach02 = np.load('C:\MPhys\\Data\\Intra Patient\\Stomach\\PCA\\EVR_Stomach02.npy') 
stomach04 = np.load('C:\MPhys\\Data\\Intra Patient\\Stomach\\PCA\\EVR_Stomach04.npy') 
stomach05 = np.load('C:\MPhys\\Data\\Intra Patient\\Stomach\\PCA\\EVR_Stomach05.npy') 
#stomach06 = np.load('C:\MPhys\\Data\\Intra Patient\\Stomach\\PCA\\EVR_Stomach06.npy') 
#stomach07 = np.load('C:\MPhys\\Data\\Intra Patient\\Stomach\\PCA\\EVR_Stomach07.npy') 

# x component
pcomponents = np.linspace(1,9,9)

plt.figure(1)
sns.set(style = "whitegrid")
plt.plot(pcomponents, panc01, 'o-', markersize = 5, clip_on = False)
plt.plot(pcomponents, stomach02, 'o-', markersize = 5, clip_on = False)
plt.plot(pcomponents, stomach04, 'o-', markersize = 5, clip_on = False)
plt.plot(pcomponents, stomach05, 'o-', markersize = 5, clip_on = False)
#plt.plot(pcomponents, stomach06, markersize = 5, clip_on = False)
#plt.plot(pcomponents, stomach07, markersize = 5, clip_on = False)

plt.title('Percentage Variance for each Principle Component',fontsize = 16)
plt.xlabel('Principal Component', fontsize = 16)
plt.ylabel('Percentage of total variance', fontsize = 16)
plt.ylim(0,1.0)
plt.xlim(1,9)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.grid(True)
