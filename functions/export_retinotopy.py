#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 15:23:50 2023

@author: daisuke
"""

import os
os.chdir('/home/daisuke/Documents/git/VariabilityEarlyVisualCortex');

import numpy as np
import os.path as osp
import scipy
import matplotlib.pyplot as plt
#import tensorflow as tf
import nibabel as nib

import functions.dstools as dst
from scipy.io import savemat
#import functions.entools3D as et3
#tf.disable_v2_behavior()

rootDir = '/home/daisuke/Documents/git/VariabilityEarlyVisualCortex';
loadDir = osp.join(rootDir,'data');
saveDir = osp.join(rootDir, 'results');
stdSphere = nib.load(osp.join(loadDir,'S1200_7T_Retinotopy181.L.sphere.32k_fs_LR.surf.gii'));

#subject_id= '157336';#'146735'

# x-y grid for 2D matrix representation
#grid_x, grid_y = np.mgrid[40:120, 0:60] #dorsal only
grid_x, grid_y = np.mgrid[20:120, 0:100] #dorsal + ventral
     
final_mask_L = dst.getFinalMask_L(False);#1D index
final_mask_L_mat = ~np.isnan(dst.gifti2mat(stdSphere, final_mask_L, final_mask_L, grid_x, grid_y));
final_mask_L_idx = dst.mat2idx(final_mask_L_mat);

final_mask_L_d = dst.getFinalMask_L(True);#1D index
final_mask_L_d_mat = ~np.isnan(dst.gifti2mat(stdSphere, final_mask_L_d, final_mask_L_d, grid_x, grid_y));
final_mask_L_d_idx = dst.mat2idx(final_mask_L_d_mat);

mask_dummy = np.ones((final_mask_L.shape)); #all one


# load polar angle data, convert to gifti space, then to 2D matrix
data = scipy.io.loadmat(osp.join(loadDir, 'cifti_polarAngle_all.mat'))['cifti_polarAngle']
data_ecc = scipy.io.loadmat(osp.join(loadDir, 'cifti_eccentricity_all.mat'))['cifti_eccentricity']

field_names = data.dtype.names;
import re
pattern = r".*_fit1_polarangle_msmall$"
filtered_strings = tuple(s for s in field_names if re.match(pattern, s))

# Initialize an empty list to store extracted parts of strings
subject_id_all =[];
pattern = r".(\d+)_fit1_polarangle_msmall$"

# obtain subject IDs
for string in filtered_strings:
    # Use re.search() to find the pattern match in the string
    match = re.search(pattern, string)
    
    # If a match is found, extract the matched part and append to the list
    if match:
        subject_id_all.append(match.group(1))

PAData_mat = np.zeros((100,100,len(subject_id_all))) #[0-360] [deg]
ECCData_mat = np.zeros((100,100,len(subject_id_all)))
for ss in range(0,len(subject_id_all)):
    subject_id = subject_id_all[ss]
    PAData = data['x' + str(subject_id) + '_fit1_polarangle_msmall'][0][0]
    PAData_L = dst.getciftiIngifti(PAData, final_mask_L);
    PAData_mat[:,:,ss] = dst.gifti2mat(stdSphere, PAData_L, final_mask_L, grid_x, grid_y);

    ECCData = data_ecc['x' + str(subject_id) + '_fit1_eccentricity_msmall'][0][0]
    ECCData_L = dst.getciftiIngifti(ECCData, final_mask_L);
    ECCData_mat[:,:,ss] = dst.gifti2mat(stdSphere, ECCData_L, final_mask_L, grid_x, grid_y);
    

# gnd avg across subjects
#import cmath
mPA_c = np.nanmean(np.exp(complex(0,1) * PAData_mat / 360 * 2*np.pi), 2)
mPAData = np.arctan2(np.imag(mPA_c), np.real(mPA_c)) * np.pi/180
mECCData = np.nanmean(ECCData_mat, 2);
plt.imshow(mPAData, extent=[np.min(grid_x),np.max(grid_x)+1, np.min(grid_y), np.max(grid_y)+1],
            origin='lower', cmap='viridis'); plt.colorbar();    
plt.show();

# convert to alititude and azimuth
mazimuth, maltitude = dst.polar_to_cartesian(mECCData, 180/np.pi*mPAData)


# obtain VFS map
theta = dst.computeFieldSign(mPAData, mECCData, smoothing = True, binarizing = False)

# Plotting the visual field sign
# if dorsal_only == True:
#     plt.imshow(theta[:,:].T, extent=[40,120, 0,60], origin='lower', cmap='viridis')
#if dorsal_only == False:
plt.imshow(theta[:,:].T, extent=[20,120, 0,100], origin='lower', cmap='viridis')


vfsfilename = os.path.join(saveDir, 'fieldSign_avg_smoothed.mat')
savemat(vfsfilename, {'vfs': theta, 'grid_z0_PA': mPAData, 'grid_z0_ecc': mECCData,
                                'mazimuth': mazimuth, 'maltitude': maltitude,
                                'grid_x': grid_x,'grid_y': grid_y, 
                                'final_mask_L': final_mask_L, 'final_mask_L_d': final_mask_L_d,
                                'final_mask_L_d_idx': final_mask_L_d_idx, 
                                'final_mask_L_idx': final_mask_L_idx});





