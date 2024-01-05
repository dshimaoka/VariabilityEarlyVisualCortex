#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 15:23:50 2023

@author: daisuke
"""

import numpy as np
import os.path as osp
import scipy
import matplotlib.pyplot as plt
import tensorflow as tf
import nibabel as nib

import functions.dstools as dst
import functions.entools3D as et3
#tf.disable_v2_behavior()

rootDir = '/home/daisuke/Documents/git/VariabilityEarlyVisualCortex';
loadDir = osp.join(rootDir,'data');
saveDir = osp.join(rootDir, 'results');
stdSphere = nib.load(osp.join(loadDir,'S1200_7T_Retinotopy181.L.sphere.32k_fs_LR.surf.gii'));

subject_id= '157336';#'146735'

# x-y grid for 2D matrix representation
#grid_x, grid_y = np.mgrid[40:120, 0:60] #dorsal only
grid_x, grid_y = np.mgrid[20:120, 0:100] #dorsal + ventral
     
final_mask_L = dst.getFinalMask_L(False);#1D index

final_mask_L_d = dst.getFinalMask_L(True);#1D index
areaMatrix = ~np.isnan(dst.gifti2mat(stdSphere, final_mask_L_d, final_mask_L_d, grid_x, grid_y));
areaIdx = dst.mat2idx(areaMatrix);

mask_dummy = np.ones((final_mask_L.shape)); #all one


# load polar angle data, convert to gifti space, then to 2D matrix
data = scipy.io.loadmat(osp.join(loadDir, 'cifti_polarAngle_all.mat'))['cifti_polarAngle']
PAData = data['x' + str(subject_id) + '_fit1_polarangle_msmall'][0][0]
PAData_L = dst.getciftiIngifti(PAData, final_mask_L);
PAData_mat = dst.gifti2mat(stdSphere, PAData_L, final_mask_L, grid_x, grid_y);
# plt.imshow(PAData_mat, extent=[np.min(grid_x),np.max(grid_x)+1, np.min(grid_y), np.max(grid_y)+1],
#            origin='lower', cmap='viridis')    
# plt.show();

# load eccentricity data, convert to gifti space, then to 2D matrix
data = scipy.io.loadmat(osp.join(loadDir, 'cifti_eccentricity_all.mat'))['cifti_eccentricity']
ECCData = data['x' + str(subject_id) + '_fit1_eccentricity_msmall'][0][0]
ECCData_L = dst.getciftiIngifti(ECCData, final_mask_L);
ECCData_mat = dst.gifti2mat(stdSphere, ECCData_L, final_mask_L, grid_x, grid_y);
plt.imshow(ECCData_mat, extent=[np.min(grid_x),np.max(grid_x)+1, np.min(grid_y), np.max(grid_y)+1],
           origin='lower', cmap='viridis')    
plt.show();

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
savemat('array_3d.mat', {'array_3d': locData_mat,'grid_x': grid_x,'grid_y': grid_y});


# curvature data projected to std surface
data = scipy.io.loadmat(osp.join(loadDir, 'cifti_curvature_all.mat'))['cifti_curvature']
curvData = data['x' + str(subject_id) + '_curvature'][0][0]
curvData_mat = dst.gifti2mat(stdSphere, dst.getciftiIngifti(curvData, final_mask_L), final_mask_L, grid_x, grid_y);
plt.imshow(curvData_mat.T, extent=[20,120, 0,100], origin='lower', cmap='viridis')
plt.show();


# load curvature from individual gifti for sanity check
#indSurf = nib.load(osp.join(loadDir,''.join([str(subject_id), '.lh.midthickness.32k_fs_LR.surf.gii'])));
#NG: we need a sphere 
# curvData2 = scipy.io.loadmat(osp.join(loadDir, ''.join(['gifti_curvature_L_', str(subject_id)])))['curvature'];
# curvData_mat2 = dst.gifti2mat(indSurf, dst.getciftiIngifti(curvData2, mask_dummy), final_mask_L, grid_x, grid_y);
# plt.imshow(curvData_mat2.T, extent=[20,120, 0,100], origin='lower', cmap='viridis')
# plt.show();


# load preferred altitude & azimuth on 2D matrix?

# load areal boundary, processed in matlab
areaMatrix = scipy.io.loadmat(osp.join(saveDir, 'fieldSign_' + str(subject_id) +  '_smoothed_arealBorder.mat'))['areaMatrix'][0]
plt.imshow(areaMatrix[0].T, extent=[20,120, 0,100], origin='lower', cmap='viridis')#V1


# sample eccentricity/PA values within V1
v1Idx = areaMatrix[0]==1;
ECCv1 = ECCData_mat[v1Idx];
PAv1 = PAData_mat[v1Idx];

xv1, yv1 = dst.polar_to_cartesian(ECCv1, 180/np.pi*PAv1)

v2v3Idx = ((areaMatrix[1]==1).astype(int)+(areaMatrix[2]==1).astype(int)).astype(bool)
ECCv2v3 = ECCData_mat[v2v3Idx];
PAv2v3 = PAData_mat[v2v3Idx];


xv1, yv1 = dst.polar_to_cartesian(ECCv1, 180/np.pi*PAv1)
xv2v3, yv2v3 = dst.polar_to_cartesian(ECCv2v3, 180/np.pi*PAv2v3)


###########################################
#### below from elastic_sweep_slurm.py ####
###########################################

# annealing parameter
#kappa = tf.placeholder(tf.float64, shape=(), name="k") #deprecated in tensorflow 2
kappa =  np.array(3.14, dtype=np.float64)

#### prototypes
# generate prototypes on the visual field - created from the real values in v2/v3
x0 = np.zeros((len(xv2v3), 2));
x0[:,0] = xv2v3;
x0[:,1] = yv2v3;

x = tf.constant(
    x0,
    dtype=tf.float64,
    name='x')

# generate prototypes on the boundary??

#### train these points on a cortical map
y0 = et3.initial_condition(v2v3Idx, x0)
y = tf.Variable(
    name = "y",
    dtype = tf.float64,
    initial_value = y0);
    
#### main cost
yx_diff = tf.expand_dims(y, 1) - tf.expand_dims(x, 0)
yx_normsq = tf.einsum('ijk,ijk->ij', yx_diff, yx_diff)
yx_gauss = tf.exp(-1.0 * yx_normsq / (2.0 * kappa * kappa))
yx_cost = -1.0 * kappa * tf.reduce_sum(
            tf.math.log(
                tf.reduce_sum(yx_gauss, axis=0)))

#### reulzrization term 1 - within area
n = et3.neighborhood(final_mask_L.astype(int))
map_h, map_w = final_mask_L.shape
mask = tf.constant(et3.make_mask(map_h, map_w, n), dtype=tf.float64, name='mask')

# pairwise distance in the visual field: first use broadcast to calculate pairwise difference
yy_diff = tf.expand_dims(y, 1) - tf.expand_dims(y, 0)
yy_normsq = tf.einsum('ijk,ijk->ij', yy_diff, yy_diff)
yy_normsq_masked = tf.multiply(mask, yy_normsq)

reg1 = tf.reduce_sum(yy_normsq_masked)






