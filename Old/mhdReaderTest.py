# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 12:58:11 2018

@author: Edward Henderson
"""

import SimpleITK as sitk
import numpy as np

def load_itk(filename):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)

    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return ct_scan, origin, spacing


#load_itk(file)


mhd = sitk.ReadImage('C:\MPhys\Data\POPI-model\\4DCT-MetaImage\\00-P.mhd')
# write out as a nifti