#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 14:17:32 2023

@author: daisuke
"""


import os
os.chdir('/home/daisuke/Documents/git/VariabilityEarlyVisualCortex');

import numpy as np
import os.path as osp

import scipy
import matplotlib.pyplot as plt
import nibabel as nib

import functions.dstools as dst
from scipy.io import savemat


doAvg = False;
showFig = False;

loadDir = '/home/daisuke/Documents/git/VariabilityEarlyVisualCortex/data/';
#saveDir = osp.join(loadDir, 'results');
saveDir = '/mnt/dshi0006_market/VariabilityEarlyVisualCortex/';
stdSphere = nib.load(osp.join(loadDir,'S1200_7T_Retinotopy181.L.sphere.32k_fs_LR.surf.gii'));

subject_id_all = dst.getSubjectId(loadDir+'cifti_polarAngle_all.mat')
#subject_id_all = ['114823','157336','585256','114823','581450','725751']; #from Ribeiro 2023 Fig1

#ng azimuth in 100610

if doAvg:
    PAData_mat_all = np.zeros((100,100,len(subject_id_all)));
    ECCData_mat_all = np.zeros((100,100,len(subject_id_all)));
    CURVData_mat_all = np.zeros((100,100,len(subject_id_all)));
    


for ids in range(0,len(subject_id_all)):
    subject_id = subject_id_all[ids]
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
    PAData_L = dst.getciftiIngifti(PAData, final_mask_L);
    PAData_mat = dst.gifti2mat(stdSphere, PAData_L, final_mask_L, grid_x, grid_y); #[deg]
   
    if showFig:
        plt.imshow(PAData_mat, extent=[np.min(grid_x),np.max(grid_x)+1, np.min(grid_y), np.max(grid_y)+1],
                    origin='lower', vmin=0, vmax=361, cmap='gist_rainbow_r'); plt.colorbar()    
        plt.show();

    
    # load eccentricity data, convert to gifti space, then to 2D matrix
    data = scipy.io.loadmat(osp.join(loadDir, 'cifti_eccentricity_all.mat'))['cifti_eccentricity']
    ECCData = data['x' + str(subject_id) + '_fit1_eccentricity_msmall'][0][0]
    ECCData_L = dst.getciftiIngifti(ECCData, final_mask_L);
    ECCData_mat = dst.gifti2mat(stdSphere, ECCData_L, final_mask_L, grid_x, grid_y);
    
    if showFig:
        plt.imshow(ECCData_mat, extent=[np.min(grid_x),np.max(grid_x)+1, np.min(grid_y), np.max(grid_y)+1],
                   origin='lower', cmap='viridis')    
        plt.show();
    
    #obtain VFS map
    #theta = dst.computeFieldSign(PAData_mat, ECCData_mat, smoothing = True, binarizing = False)
    #plt.imshow(theta[:,:], extent=[20,120, 0,100], origin='lower', cmap='viridis')
    
    # convert to alititude and azimuth [deg]
    #azimuth, altitude = dst.polar_to_cartesian(ECCData_mat, np.pi/180*PAData_mat) NG for azimuth 
    azimuth_L, altitude_L = dst.polar_to_cartesian(ECCData_L, np.pi/180*PAData_L) 
    azimuth = dst.gifti2mat(stdSphere, azimuth_L, final_mask_L, grid_x, grid_y); #[deg]
    altitude = dst.gifti2mat(stdSphere, altitude_L, final_mask_L, grid_x, grid_y); #[deg]
    
    
    # load curvature data (used for sanity check in compute_minimal_path_femesh_individual), convert to gifti space, then to 2D matrix
    data = scipy.io.loadmat(osp.join(loadDir, 'cifti_curvature_all.mat'))['cifti_curvature']
    CURVData = data['x' + str(subject_id) + '_curvature'][0][0]
    CURVData_L = dst.getciftiIngifti(CURVData, final_mask_L);
    CURVData_mat = dst.gifti2mat(stdSphere, CURVData_L, final_mask_L, grid_x, grid_y);

    if showFig: 
        plt.imshow(CURVData_mat, extent=[np.min(grid_x),np.max(grid_x)+1, np.min(grid_y), np.max(grid_y)+1],
                   origin='lower')
        plt.show();
    
    # obtain 1D idx of the matrix
    #idx = dst.matIdx(matrix, final_mask_L);
    #matrix = dst.gifti2mat(stdSphere, final_mask_L, final_mask_L, grid_x, grid_y);
    
    
    # load 3D geometrical data of individual brain, converto gifti space, then to 2D matrix
    locData_mat = np.zeros((len(grid_y),len(grid_x),3))
    if doAvg:
        locData = scipy.io.loadmat(osp.join(loadDir, 'mid_pos_L.mat'))['mid_pos_L'] #grand average
    else:
        thisDir = osp.join(saveDir, subject_id);
        locData = scipy.io.loadmat(osp.join(thisDir, 'mid_pos_L_' + subject_id + '.mat'))['mid_pos_L'] # data from Fernanda through RDS
        
    
    #locData = data['pos'][0][0]; #sphere
    for idim in range(0,3):
        locData_tmp = dst.getciftiIngifti(locData[:,idim], final_mask_L);
        locData_mat[:,:,idim] = dst.gifti2mat(stdSphere, locData_tmp, final_mask_L, grid_x, grid_y);

    if showFig:    
        plt.imshow(locData_mat[:,:,1], extent=[np.min(grid_x),np.max(grid_x)+1, np.min(grid_y), np.max(grid_y)+1],
                   origin='lower', cmap='viridis')    
        plt.show();
    
    
    if doAvg == False:
        savemat(osp.join(thisDir, 'geometry_retinotopy_' + subject_id + '.mat'), 
                {'array_3d': locData_mat,'grid_x': grid_x,'grid_y': grid_y, 
                  'grid_azimuth': azimuth, 'grid_altitude': altitude, 'grid_curv': CURVData_mat,
                  'final_mask_L': final_mask_L,'final_mask_L_d': final_mask_L_d,
                  'final_mask_L_d_idx': final_mask_L_d_idx, 
                  'final_mask_L_idx': final_mask_L_idx});
        # no longer saved:
        #    'grid_PA': PAData_mat, 'grid_ecc': ECCData_mat,'vfs': theta, 
            
    else:
        PAData_mat_all[:,:,ids] = PAData_mat;
        ECCData_mat_all[:,:,ids] = ECCData_mat;
        CURVData_mat_all[:,:,ids] = CURVData_mat;
        

if doAvg:        
    # gnd avg across subjects
    #import cmath
    mPA_c = np.nanmean(np.exp(complex(0,1) * PAData_mat_all / 180 * np.pi), axis=2)
    mPAData_mat = np.arctan2(np.imag(mPA_c), np.real(mPA_c))*180/np.pi #deg
    sum = mPAData_mat < 0
    mPAData_mat[sum] = mPAData_mat[sum] + 360

    mECCData_mat = np.nanmean(ECCData_mat_all, 2); #deg
    mCURVData_mat = np.nanmean(CURVData_mat_all, 2);
        
    # convert to alititude and azimuth [deg]
    mazimuth, maltitude = dst.polar_to_cartesian(mECCData_mat, np.pi/180*mPAData_mat) 
    
    # obtain VFS map
    theta = dst.computeFieldSign(mPAData_mat, mECCData_mat, smoothing = True, binarizing = False)

    thisDir = osp.join(saveDir, 'avg');
    savemat(osp.join(thisDir, 'geometry_retinotopy_avg.mat'), 
            {'array_3d': locData_mat,'grid_x': grid_x,'grid_y': grid_y, 
              'grid_azimuth': mazimuth, 'grid_altitude': maltitude, 'grid_curv': mCURVData_mat,
              'final_mask_L': final_mask_L,'final_mask_L_d': final_mask_L_d,
              'final_mask_L_d_idx': final_mask_L_d_idx, 
              'final_mask_L_idx': final_mask_L_idx});