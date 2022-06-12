import numpy as np
import scipy.io
import os.path as osp

from functions.def_ROIs_DorsalEarlyVisualCortex import roi
from functions.def_ROIs_EarlyVisualAreas import roi as ROI
from nilearn import plotting

def polarAngle_plot(subject_id, hemisphere, mode):
    path = './../data'
    curv = scipy.io.loadmat(osp.join(path, 'cifti_curvature_all.mat'))[
        'cifti_curv']
    background = np.reshape(
        curv['x' + subject_id + '_curvature'][0][0][0:32492], (-1))

    threshold = 1  # threshold for the curvature map

    # Background settings
    nocurv = np.isnan(background)
    background[nocurv == 1] = 0
    background[background < 0] = 0
    background[background > 0] = 1

    # Mask - Visual cortex
    final_mask_L_ROI, final_mask_R_ROI, index_L_mask_ROI, index_R_mask_ROI = \
        ROI(['ROI'])

    # Mask - Early visual cortex
    final_mask_L, final_mask_R, index_L_mask, index_R_mask = \
        roi(['ROI'])

    # Mask for eccentricity range
    eccentricity_mask_LH = np.reshape(
        np.load('./../main/MaskEccentricity_'
                'above1below8ecc_LH.npz')['list'], (-1))

    # Final mask
    mask_LH = final_mask_L_ROI + final_mask_L
    mask_LH[mask_LH != 2] = 0
    mask_LH[mask_LH == 2] = 1
    mask_LH = mask_LH[final_mask_L_ROI == 1]  # * eccentricity_mask_LH


    # final mask
    final_mask_L_ROI[final_mask_L_ROI==1]=mask_LH

    polarAngle = np.zeros((32492, 1))

    number_cortical_nodes = int(64984)
    number_hemi_nodes = int(number_cortical_nodes / 2)

    # Loading the predictions
    data = np.load('./../output/cluster_'+ str(mode) +'_weightedJaccard_eccentricityMask.npz')['list']

    if hemisphere == 'left':

        polarAngle[final_mask_L_ROI == 1] = np.reshape(
            data, (-1, 1))

        # Masking
        pred = np.array(polarAngle) + threshold
        pred[final_mask_L != 1] = 0

        # # Binarizing shifted values
        # pred[(pred >= 180) & (pred <= 225)] = 0
        # pred[(pred > 225) & (pred <= 360)] = 90
        # pred[(pred >= 135) & (pred < 180)] = 360
        # pred[(pred >= 0) & (pred < 135)] = 270
        # pred[final_mask_L_ROI != 1] = 0


        # Reshifting
        pred = np.array(pred)
        minus = pred > 180
        sum = pred < 180
        pred[minus] = pred[minus] - 180 + threshold
        pred[sum] = pred[sum] + 180 + threshold
        pred[final_mask_L != 1] = 0

        # Binarizing reshifted values
        pred[(pred >= 0) & (pred <= 45)] = 0 + threshold
        pred[(pred > 45) & (pred <= 180)]= 90 + threshold
        pred[(pred >= 315) & (pred <= 360)] = 360 + threshold
        pred[(pred > 180) & (pred < 315)] = 270 + threshold
        pred[final_mask_L != 1] = 0


        # Empirical map
        view = plotting.view_surf(
            surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '../data'
                 '/S1200_7T_Retinotopy181.L.sphere.32k_fs_LR.surf.gii'),
            surf_map=np.reshape(pred[0:32492], (-1)), bg_map=background,
            cmap='gist_rainbow_r', black_bg=False, symmetric_cmap=False,
            threshold=threshold, vmax=361)
        view.open_in_browser()

    if hemisphere == 'right':
        polarAngle[final_mask_R == 1] = np.reshape(
            data['x' + str(subject_id) + '_fit1_polarangle_msmall'][0][0][
            number_hemi_nodes:number_cortical_nodes].reshape(
                (number_hemi_nodes))[final_mask_R == 1], (-1, 1))

        # Masking
        pred = np.array(polarAngle) + threshold
        pred[final_mask_R != 1] = 0

        # Empirical map
        view = plotting.view_surf(
            surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '../data'
                 '/S1200_7T_Retinotopy181.R.sphere.32k_fs_LR.surf.gii'),
            surf_map=np.reshape(pred[0:32492], (-1)), bg_map=background,
            cmap='gist_rainbow', black_bg=False, symmetric_cmap=False,
            threshold=threshold, vmax=361)
        view.open_in_browser()

subject = '111312'
hemisphere = 'left'

for i in range(7):
    polarAngle_plot(subject, hemisphere, mode = i)
