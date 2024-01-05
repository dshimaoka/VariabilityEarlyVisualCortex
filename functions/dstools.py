#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created from gradientAnalysis.py

@author: daisuke
"""

#import os.path as osp
import numpy as np


from scipy.interpolate import griddata
from functions.def_ROIs_EarlyVisualAreas import roi
from functions.def_ROIs_DorsalEarlyVisualCortex import roi as ROI
from scipy.ndimage import gaussian_filter

number_cortical_nodes = int(64984)
number_hemi_nodes = int(number_cortical_nodes / 2)

def ind2sub(shape, index):
    # Convert subscripts to linear index
        
    num_columns = shape[1]
    row_index = np.floor(index / num_columns)
    column_index = index - row_index * num_columns
    return row_index, column_index

def sub2ind(shape, row_index, column_index):
    # Convert subscripts to linear index
        
    num_columns = shape[1]
    index = row_index * num_columns + column_index
    return index


def getciftiIngifti(ciftiData, final_mask_L):

 final_mask_L_dorsal = final_mask_L;
 
 # Number of nodes
 number_cortical_nodes = int(64984)
 number_hemi_nodes = int(number_cortical_nodes / 2)

 # Loading the flat surface 32k gifti
 # flat_surf = nib.load(osp.join(path,'S1200_7T_Retinotopy181.L.sphere.32k_fs_LR.surf.gii'))
 # flat_surf_pos = flat_surf.agg_data('pointset')[final_mask_L*final_mask_L_dorsal==1]
 # coord_plane = np.array(flat_surf_pos.T[0:3,].T).astype(int)
 # new_coord_plane = np.matmul([[0, 0, 1],[0, 1, 0]], coord_plane.T).T + 100


 #### Polar angle map ####
 # Loading polar angle values
 data = np.zeros((number_hemi_nodes, 1))
 data[final_mask_L*final_mask_L_dorsal == 1] = np.reshape(
             ciftiData[0:number_hemi_nodes].reshape(
                     (number_hemi_nodes))[final_mask_L*final_mask_L_dorsal == 1], (-1, 1)) 
 return data


def gifti2mat(flat_surf, giftiData, final_mask_L, grid_x, grid_y):

  final_mask_L_dorsal = final_mask_L;
 
  #flat_surf = nib.load(osp.join(loadDir,'S1200_7T_Retinotopy181.L.sphere.32k_fs_LR.surf.gii'));

  flat_surf_pos = flat_surf.agg_data('pointset')[final_mask_L*final_mask_L_dorsal==1]
  coord_plane = np.array(flat_surf_pos.T[0:3,].T).astype(int)
  new_coord_plane = np.matmul([[0, 0, 1],[0, 1, 0]], coord_plane.T).T + 100

  matrixData = griddata(new_coord_plane, giftiData[final_mask_L*final_mask_L_dorsal == 1], 
                        (grid_x, grid_y), method='linear')
  
  matrixData = np.squeeze(matrixData);
  
  return matrixData


def idx2mat(linear_idx, grid_x, grid_y):
    
    areaMatrix = np.zeros((len(grid_y)*len(grid_x),1))    
    areaMatrix[linear_idx.astype(int)] = 1;
    areaMatrix = np.reshape( areaMatrix, (len(grid_y), len(grid_x)))
    return  areaMatrix


def mat2idx(areaMatrix):
    subscripts2D = np.where(areaMatrix.astype(int) == 1)
    # Zip the subscripts to get pairs (row, column)
    row_indices, column_indices = subscripts2D
    subscripts1D = list(zip(row_indices, column_indices))

    # Convert subscripts to linear index
    linear_idx = np.zeros((len(subscripts1D),1))
    for ii in range(len(subscripts1D)):
        linear_idx[ii] = np.ravel_multi_index(subscripts1D[ii], dims=areaMatrix.shape, order='C')

    return linear_idx

# def matIdx(matrix, matrix_mask = None):
#     # FIXME 2ND INPUT
    
#     #num_rows = len(matrix)
#     num_cols = len(matrix[0])

#     row_index = 1
#     col_index = 2

#     # Get the 1D index
#     index_1d = np.arange(0, get_1d_index(row_index, col_index, num_cols)-1)
#     return index_1d
    

# def get_1d_index(row, col, num_cols):
#     return row * num_cols + col

# def idx2sub(index_1d, matrik_mask):
#     # FIXME
#     return sub

def getFinalMask_L(dorsal_only=True):
   #FIXEME SO IT APPLIES TO EVERY BRAIN
   #CURRENTLY THIS APPLIES TO ONLY THE STANDARD SURFACE
   
   # Early visual cortex
   label_primary_visual_areas = ['ROI']
   final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
       label_primary_visual_areas)

   if dorsal_only == True:
       label_primary_visual_areas = ['ROI']
       final_mask_L_dorsal, final_mask_R_dorsal, index_L_mask_dorsal, index_R_mask_dorsal = ROI(
           label_primary_visual_areas)
       final_mask_L = final_mask_L * final_mask_L_dorsal
       
   return final_mask_L

def computeFieldSign(grid_z0_PA, grid_z0_ecc, smoothing = True, binarizing = True):
    #low level function to compute field sign
    # grid_z0_PA: polar angle, represented in 2D matrix
    # grid_z0_ecc: eccentricity, represented in 2D matrix 
    # Smoothing the polar angle map
    if smoothing == True:
        grid_z0_PA = gaussian_filter(grid_z0_PA, sigma=.8)
    
    # Determining the gradient
    dx_PA, dy_PA = np.gradient(grid_z0_PA)
    dx_ecc, dy_ecc = np.gradient(grid_z0_ecc)

    #### Field sign analysis ####
    # Angle between gradient vectors
    dot_product = dx_PA*dx_ecc + dy_PA*dy_ecc
    modulus_PA = np.sqrt(dx_PA**2 + dy_PA**2)
    modulus_ecc = np.sqrt(dx_ecc**2 + dy_ecc**2)
    theta = np.arccos(dot_product/(modulus_PA*modulus_ecc))
    
    # Cross product
    cross_product = dx_PA*dy_ecc - dy_PA*dx_ecc

    # Binarizing
    theta[cross_product<0] = 2*np.pi - theta[cross_product<0]
    if binarizing == True:
         theta[theta>np.pi] = 2*np.pi
         theta[theta<np.pi] = np.pi

    theta = (theta - np.pi) / np.pi
    
    return theta

def polar_to_cartesian(r, theta):
    #theta is in radian
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y