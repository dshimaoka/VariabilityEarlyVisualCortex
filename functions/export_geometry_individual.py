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
from scipy.io import savemat

rootDir = '/home/daisuke/Documents/git/VariabilityEarlyVisualCortex';
loadDir = osp.join(rootDir,'data');
saveDir = osp.join(rootDir, 'results');
stdSphere = nib.load(osp.join(loadDir,'S1200_7T_Retinotopy181.L.sphere.32k_fs_LR.surf.gii'));

#subject_id= '157336';#'146735'
all_ids = ['114823','157336','585256','114823','581450','725751'];

for ids in range(0,len(all_ids)):
    subject_id = all_ids[ids]
    # x-y grid for 2D matrix representation
    #grid_x, grid_y = np.mgrid[40:120, 0:60] #dorsal only
    grid_x, grid_y = np.mgrid[20:120, 0:100] #dorsal + ventral
         
    final_mask_L = dst.getFinalMask_L(False);#1D index
    final_mask_L_mat = ~np.isnan(dst.gifti2mat(stdSphere, final_mask_L, final_mask_L, grid_x, grid_y));
    final_mask_L_idx = dst.mat2idx(final_mask_L_mat);
    
    final_mask_L_d = dst.getFinalMask_L(True);#1D index
    final_mask_L_d_mat = ~np.isnan(dst.gifti2mat(stdSphere, final_mask_L_d, final_mask_L_d, grid_x, grid_y));
    final_mask_L_d_idx = dst.mat2idx(final_mask_L_d_mat);
    
    
    areaMatrix = ~np.isnan(dst.gifti2mat(stdSphere, final_mask_L_d, final_mask_L_d, grid_x, grid_y));
    areaIdx = dst.mat2idx(areaMatrix);
    
    mask_dummy = np.ones((final_mask_L.shape)); #all one
    
    
    # load polar angle data, convert to gifti space, then to 2D matrix
    data = scipy.io.loadmat(osp.join(loadDir, 'cifti_polarAngle_all.mat'))['cifti_polarAngle']
    PAData = data['x' + str(subject_id) + '_fit1_polarangle_msmall'][0][0]
    sum = PAData < 180
    minus = PAData > 180
    PAData[sum] = PAData[sum] + 180
    PAData[minus] = PAData[minus] - 180
    PAData_L = dst.getciftiIngifti(PAData, final_mask_L);
    PAData_mat = np.pi/180*dst.gifti2mat(stdSphere, PAData_L, final_mask_L, grid_x, grid_y); #[radian]
   
    plt.imshow(PAData_mat, extent=[np.min(grid_x),np.max(grid_x)+1, np.min(grid_y), np.max(grid_y)+1],
                origin='lower', cmap='viridis'); plt.colorbar()    
    plt.show();
    
    # load eccentricity data, convert to gifti space, then to 2D matrix
    data = scipy.io.loadmat(osp.join(loadDir, 'cifti_eccentricity_all.mat'))['cifti_eccentricity']
    ECCData = data['x' + str(subject_id) + '_fit1_eccentricity_msmall'][0][0]
    ECCData_L = dst.getciftiIngifti(ECCData, final_mask_L);
    ECCData_mat = dst.gifti2mat(stdSphere, ECCData_L, final_mask_L, grid_x, grid_y);
    plt.imshow(ECCData_mat, extent=[np.min(grid_x),np.max(grid_x)+1, np.min(grid_y), np.max(grid_y)+1],
                origin='lower', cmap='viridis')    
    plt.show();
    
    #obtain VFS map
    theta = dst.computeFieldSign(PAData_mat, ECCData_mat, smoothing = True, binarizing = False)
    plt.imshow(theta[:,:], extent=[20,120, 0,100], origin='lower', cmap='viridis')
    
    # convert to alititude and azimuth
    azimuth, altitude = dst.polar_to_cartesian(ECCData_mat, PAData_mat)
    
    
    # obtain 1D idx of the matrix
    #idx = dst.matIdx(matrix, final_mask_L);
    #matrix = dst.gifti2mat(stdSphere, final_mask_L, final_mask_L, grid_x, grid_y);
    
    
    # load 3D geometrical data of individual brain, converto gifti space, then to 2D matrix
    # data from Fernanda through RDS
    locData_mat = np.zeros((len(grid_y),len(grid_x),3))
    locData = scipy.io.loadmat(osp.join(loadDir, 'mid_pos_L_' + subject_id + '.mat'))['mid_pos_L']
    #locData = data['pos'][0][0]; #sphere
    for idim in range(0,3):
        locData_tmp = dst.getciftiIngifti(locData[:,idim], final_mask_L);
        locData_mat[:,:,idim] = dst.gifti2mat(stdSphere, locData_tmp, final_mask_L, grid_x, grid_y);
    
    plt.imshow(locData_mat[:,:,1], extent=[np.min(grid_x),np.max(grid_x)+1, np.min(grid_y), np.max(grid_y)+1],
               origin='lower', cmap='viridis')    
    plt.show();
    
    
    savemat([saveDir + 'geometry_retinotopy_' + subject_id + '.mat'], 
            {'array_3d': locData_mat,'grid_x': grid_x,'grid_y': grid_y, 
             'grid_PA': PAData_mat, 'grid_ecc': ECCData_mat,
             'grid_azimuth': azimuth, 'grid_altitude': altitude,
             'vfs': theta,
             'final_mask_L': final_mask_L,'final_mask_L_d': final_mask_L_d,
             'final_mask_L_d_idx': final_mask_L_d_idx, 
             'final_mask_L_idx': final_mask_L_idx});
    
    # {'vfs': theta, 'grid_z0_PA': mPAData, 'grid_z0_ecc': mECCData,
    #                                'mazimuth': mazimuth, 'maltitude': maltitude,
    #                                'grid_x': grid_x,'grid_y': grid_y, 
    #                                'final_mask_L': final_mask_L, 'final_mask_L_d': final_mask_L_d,
    #                                'final_mask_L_d_idx': final_mask_L_d_idx, 
    #                                'final_mask_L_idx': final_mask_L_idx});
