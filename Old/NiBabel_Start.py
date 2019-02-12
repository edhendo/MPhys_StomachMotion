# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 17:05:28 2018

@author: Edward Henderson
"""

# nibabel example code
import os
import numpy as np
from nibabel.testing import data_path
np.set_printoptions(precision=2, suppress=True)
example_filename = os.path.join(data_path, 'example4d.nii.gz')

import nibabel as nib
img = nib.load(example_filename)

# NiBabel image knows about it's shape
# > img.shape
# It also records the data type of the data as stored on disk.
# In this case the data on disk are 16 bit signed integers:
# > img.get_data_dtype() == np.dtype(np.int16)
# The image has an affine transformation that determines the world-coordinates 
# of the image elements
# > img.affine.shape

data = img.get_fdata()
# > data.shape
# > type(data)

hdr = img.header
# > hdr.get_xyzt_units()
raw = hdr.structarr
# > raw['xyzt_units']

# loading/saving
imgLoad = nib.load('C:\MPhys\\Nifti_Images\\101\\averageVecs.nii')
hdr2 = imgLoad.header
data2 = imgLoad.get_fdata()