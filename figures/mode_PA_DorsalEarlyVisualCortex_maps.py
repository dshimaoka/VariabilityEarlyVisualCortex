import numpy as np
import scipy.io
import os.path as osp
import sys

sys.path.append('..')
from functions.def_ROIs_DorsalEarlyVisualCortex import roi
from functions.def_ROIs_EarlyVisualAreas import roi as ROI
from nilearn import plotting

def polarAngle_plot(subject_id, path, cluster, binarize = False):
    """
    Plot the polar angle map of the early visual cortex.
    Parameters
    ----------
    subject_id : int
        Subject ID.
    path : str  
        Path to the data.
    cluster : str
        cluster index.
    binarize : bool, optional
        Binarize the polar angle map. The default is False.
    Returns
    -------
    Plot of the mean polar angle map from the given cluster.
    """
    # Loading the curvature map
    curv = scipy.io.loadmat(osp.join(path, 'cifti_curvature_all.mat'))[
        'cifti_curv']
    background = np.reshape(
        curv['x' + subject_id + '_curvature'][0][0][0:32492], (-1))

    # Background settings
    threshold = 1  # threshold for the curvature map
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

    # # Mask for eccentricity range
    # eccentricity_mask_LH = np.reshape(
    #     np.load('./../main/MaskEccentricity_'
    #             'above1below8ecc_LH.npz')['list'], (-1))

    # Final mask
    mask_LH = final_mask_L_ROI + final_mask_L
    mask_LH[mask_LH != 2] = 0
    mask_LH[mask_LH == 2] = 1
    mask_LH = mask_LH[final_mask_L_ROI == 1]
    final_mask_L_ROI[final_mask_L_ROI==1]=mask_LH

    # Number of nodes
    number_cortical_nodes = int(64984)
    number_hemi_nodes = int(number_cortical_nodes / 2)

    # Loading the data
    polarAngle = np.zeros((32492, 1))
    data = np.load('./../output/cluster_'+ str(cluster) +'_PAmaps_weightedJaccard_eccentricityMask.npz')['list']
    polarAngle[final_mask_L_ROI == 1] = np.reshape(
        data, (-1, 1))

    # Masking
    polarAngle = np.array(polarAngle) + threshold
    polarAngle[final_mask_L != 1] = 0

    # Reshifting
    polarAngle = np.array(polarAngle)
    minus = polarAngle > 180
    sum = polarAngle < 180
    polarAngle[minus] = polarAngle[minus] - 180 + threshold
    polarAngle[sum] = polarAngle[sum] + 180 + threshold
    polarAngle[final_mask_L != 1] = 0

    # Binarizing reshifted values
    if binarize==True:
        polarAngle[(polarAngle >= 0) & (polarAngle <= 45)] = 0 + threshold
        polarAngle[(polarAngle > 45) & (polarAngle <= 180)]= 90 + threshold
        polarAngle[(polarAngle >= 315) & (polarAngle <= 360)] = 360 + threshold
        polarAngle[(polarAngle > 180) & (polarAngle < 315)] = 270 + threshold
        polarAngle[final_mask_L != 1] = 0

    # Plotting
    view = plotting.view_surf(
        surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '../data'
                '/S1200_7T_Retinotopy181.L.sphere.32k_fs_LR.surf.gii'),
        surf_map=np.reshape(polarAngle[0:32492], (-1)), bg_map=background,
        cmap='gist_rainbow_r', black_bg=False, symmetric_cmap=False,
        threshold=threshold, vmax=361)
    return view.open_in_browser()

if __name__ == '__main__':
    curv_background_subject = '111312'
    hemisphere = 'left'

    for i in range(7):
        polarAngle_plot(curv_background_subject, './../data/',cluster = i, binarize = True)
