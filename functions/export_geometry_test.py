#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 14:17:32 2023

@author: daisuke
"""



import numpy as np
import os.path as osp
import scipy
import matplotlib.pyplot as plt
import nibabel as nib

import functions.dstools as dst
#import functions.entools3D as et3

rootDir = '/home/daisuke/Documents/git/VariabilityEarlyVisualCortex';
loadDir = osp.join(rootDir,'data');
saveDir = osp.join(rootDir, 'results');
stdSphere = nib.load(osp.join(loadDir,'S1200_7T_Retinotopy181.L.sphere.32k_fs_LR.surf.gii'));

#subject_id= '157336';#'146735'

# x-y grid for 2D matrix representation
#grid_x, grid_y = np.mgrid[40:120, 0:60] #dorsal only
grid_x, grid_y = np.mgrid[20:120, 0:100] #dorsal + ventral
     
final_mask_L = dst.getFinalMask_L(False);#1D index

final_mask_L_d = dst.getFinalMask_L(True);#1D index
areaMatrix = ~np.isnan(dst.gifti2mat(stdSphere, final_mask_L_d, final_mask_L_d, grid_x, grid_y));
areaIdx = dst.mat2idx(areaMatrix);

mask_dummy = np.ones((final_mask_L.shape)); #all one


# # load polar angle data, convert to gifti space, then to 2D matrix
# data = scipy.io.loadmat(osp.join(loadDir, 'cifti_polarAngle_all.mat'))['cifti_polarAngle']
# PAData = data['x' + str(subject_id) + '_fit1_polarangle_msmall'][0][0]
# PAData_L = dst.getciftiIngifti(PAData, final_mask_L);
# PAData_mat = dst.gifti2mat(stdSphere, PAData_L, final_mask_L, grid_x, grid_y);
# plt.imshow(PAData_mat, extent=[np.min(grid_x),np.max(grid_x)+1, np.min(grid_y), np.max(grid_y)+1],
#            origin='lower', cmap='viridis')    
# plt.show();

# # load eccentricity data, convert to gifti space, then to 2D matrix
# data = scipy.io.loadmat(osp.join(loadDir, 'cifti_eccentricity_all.mat'))['cifti_eccentricity']
# ECCData = data['x' + str(subject_id) + '_fit1_eccentricity_msmall'][0][0]
# ECCData_L = dst.getciftiIngifti(ECCData, final_mask_L);
# ECCData_mat = dst.gifti2mat(stdSphere, ECCData_L, final_mask_L, grid_x, grid_y);
# plt.imshow(ECCData_mat, extent=[np.min(grid_x),np.max(grid_x)+1, np.min(grid_y), np.max(grid_y)+1],
#            origin='lower', cmap='viridis')    
# plt.show();

# obtain VFS map
#theta = dst.computeFieldSign(grid_z0_PA, grid_z0_ecc, smoothing = True, binarizing = False)

# Plotting the visual field sign
# if dorsal_only == True:
#     plt.imshow(theta[:,:].T, extent=[40,120, 0,60], origin='lower', cmap='viridis')
# if dorsal_only == False:
#     plt.imshow(theta[:,:].T, extent=[20,120, 0,100], origin='lower', cmap='viridis')


# obtain 1D idx of the matrix
#idx = dst.matIdx(matrix, final_mask_L);
#matrix = dst.gifti2mat(stdSphere, final_mask_L, final_mask_L, grid_x, grid_y);


# load 3D geometrical data, converto gifti space, then to 2D matrix
locData_mat = np.zeros((len(grid_y),len(grid_x),3))
locData = scipy.io.loadmat(osp.join(loadDir, 'mid_pos_L.mat'))['mid_pos_L']#grand average
#locData = data['pos'][0][0]; #sphere
for idim in range(0,3):
    locData_tmp = dst.getciftiIngifti(locData[:,idim], final_mask_L);
    locData_mat[:,:,idim] = dst.gifti2mat(stdSphere, locData_tmp, final_mask_L, grid_x, grid_y);
plt.imshow(locData_mat[:,:,0], extent=[np.min(grid_x),np.max(grid_x)+1, np.min(grid_y), np.max(grid_y)+1],
           origin='lower', cmap='viridis')    
plt.show();


from scipy.io import savemat
savemat('array_3d.mat', {'array_3d': locData_mat,'grid_x': grid_x,'grid_y': grid_y, 'mask': final_mask_L});
