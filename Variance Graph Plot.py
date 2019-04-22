# Variance Graph Plot
"""
Created on Tue Mar 26 16:30:54 2019

@author: Eleanor
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(style = "whitegrid")

# import all PCA data for magnitudes - y components
panc01 = np.load('C:\MPhys\\Data\\Intra Patient\\Stomach_Interpolated\\PCA\\EVR_Panc01.npy')
stomach02 = np.load('C:\MPhys\\Data\\Intra Patient\\Stomach_Interpolated\\PCA\\EVR_Stomach02.npy') 
stomach04 = np.load('C:\MPhys\\Data\\Intra Patient\\Stomach_Interpolated\\PCA\\EVR_Stomach04.npy') 
stomach05 = np.load('C:\MPhys\\Data\\Intra Patient\\Stomach_Interpolated\\PCA\\EVR_Stomach05.npy') 
stomach06 = np.load('C:\MPhys\\Data\\Intra Patient\\Stomach_Interpolated\\PCA\\EVR_Stomach06.npy') 
stomach07 = np.load('C:\MPhys\\Data\\Intra Patient\\Stomach_Interpolated\\PCA\\EVR_Stomach07.npy') 

# x component
pcomponents = np.linspace(1,9,9)
pcomponents2 = np.linspace(2,9,8)
###############################################################################################################
plt.figure(1)
plt.plot(pcomponents, panc01, 'o-', markersize = 5, clip_on = False, zorder = 10, label = 'Patient 01')
plt.plot(pcomponents, stomach02, 'o-', markersize = 5, clip_on = False, zorder = 10, label = 'Patient 02')
plt.plot(pcomponents, stomach04, 'o-', markersize = 5, clip_on = False, zorder = 10, label = 'Patient 04')
plt.plot(pcomponents, stomach05, 'o-', markersize = 5, clip_on = False, zorder = 10, label = 'Patient 05')
plt.plot(pcomponents, stomach06, 'o-', markersize = 5, clip_on = False, zorder = 10, label = 'Patient 06')
plt.plot(pcomponents, stomach07, 'o-', markersize = 5, clip_on = False, zorder = 10, label = 'Patient 07')

plt.title('Percentage Variance for each Principle Component',fontsize = 16)
plt.xlabel('Principal Component', fontsize = 16)
plt.ylabel('Percentage of total variance', fontsize = 16)
plt.ylim(0,1.0)
plt.xlim(1,9)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('C:\MPhys\\Data\\Intra Patient\\Stomach_Interpolated\\PCA Graphs and Images\\All_Patients_Variance.png')
################################################################################################################
plt.figure(2)
plt.semilogy(pcomponents, panc01, 'o-', markersize = 5, clip_on = False, zorder = 10, label = 'Patient 01')
plt.semilogy(pcomponents, stomach02, 'o-', markersize = 5, clip_on = False, zorder = 10, label = 'Patient 02')
plt.semilogy(pcomponents, stomach04, 'o-', markersize = 5, clip_on = False, zorder = 10, label = 'Patient 04')
plt.semilogy(pcomponents, stomach05, 'o-', markersize = 5, clip_on = False, zorder = 10, label = 'Patient 05')
plt.semilogy(pcomponents, stomach06, 'o-', markersize = 5, clip_on = False, zorder = 10, label = 'Patient 06')
plt.semilogy(pcomponents, stomach07, 'o-', markersize = 5, clip_on = False, zorder = 10, label = 'Patient 07')

plt.title('Percentage Variance for each Principle Component',fontsize = 16)
plt.xlabel('Principal Component', fontsize = 16)
plt.ylabel('Percentage of total variance - log scaled', fontsize = 16)
plt.ylim(0,1.0)
plt.xlim(1,9)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('C:\MPhys\\Data\\Intra Patient\\Stomach_Interpolated\\PCA Graphs and Images\\All_Patients_LogVariance.png')

plt.figure(3)
plt.plot(pcomponents2, panc01[1:], 'o-', markersize = 5, clip_on = False, zorder = 10, label = 'Patient 01')
plt.plot(pcomponents2, stomach02[1:], 'o-', markersize = 5, clip_on = False, zorder = 10, label = 'Patient 02')
plt.plot(pcomponents2, stomach04[1:], 'o-', markersize = 5, clip_on = False, zorder = 10, label = 'Patient 04')
plt.plot(pcomponents2, stomach05[1:], 'o-', markersize = 5, clip_on = False, zorder = 10, label = 'Patient 05')
plt.plot(pcomponents2, stomach06[1:], 'o-', markersize = 5, clip_on = False, zorder = 10, label = 'Patient 06')
plt.plot(pcomponents2, stomach07[1:], 'o-', markersize = 5, clip_on = False, zorder = 10, label = 'Patient 07')

plt.title('Percentage Variance for each Principle Component',fontsize = 16)
plt.xlabel('Principal Component', fontsize = 16)
plt.ylabel('Percentage of total variance', fontsize = 16)
plt.ylim(0,0.12)
plt.xlim(2,9)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('C:\MPhys\\Data\\Intra Patient\\Stomach_Interpolated\\PCA Graphs and Images\\All_Patients_Variance2.png')