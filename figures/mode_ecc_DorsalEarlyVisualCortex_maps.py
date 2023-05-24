import numpy as np
import scipy.io
import os.path as osp
import sys

sys.path.append('..')

from nilearn import plotting
from functions.def_ROIs_EarlyVisualAreas import roi as ROI
from functions.def_ROIs_DorsalEarlyVisualCortex import roi

def eccentricity_plot(subject_id, path, cluster, binarize=False):
    """
    Plot the eccentricity map of the early visual cortex.
    Parameters
    ----------
    subject_id : int
        Subject ID.
    path : str
        Path to the data.
    cluster : int
        cluster index.
    binarize : bool, optional
        Binarize the eccentricity map. The default is False.
    Returns
    -------
    Plot of the mean eccentricity map from the given cluster.
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

    # Early visual cortex
    final_mask_L_ROI, final_mask_R_ROI, index_L_mask_ROI, index_R_mask_ROI = \
        ROI(['ROI'])

    # Dorsal portion
    final_mask_L_dorsal, final_mask_R_dorsal, index_L_mask_dorsal, index_R_mask_dorsal = \
        roi(['ROI'])

    # # Mask for eccentricity range
    # eccentricity_mask_LH = np.reshape(
    #     np.load('./../main/MaskEccentricity_'
    #             'above1below8ecc_LH.npz')['list'], (-1))

    # Final mask
    mask_LH = final_mask_L_ROI + final_mask_L_dorsal
    mask_LH[mask_LH != 2] = 0
    mask_LH[mask_LH == 2] = 1
    mask_LH = mask_LH[final_mask_L_ROI == 1]
    final_mask_L_ROI[final_mask_L_ROI == 1] = mask_LH

    # Loading the data
    eccentricity = np.zeros((32492, 1))
    data = np.load('./../output/cluster_' + str(cluster) +
                   '_eccMaps_weightedJaccard_eccentricityMask.npz')['list']

    # Masking
    eccentricity[final_mask_L_ROI == 1] = np.reshape(
        data, (-1, 1))
    eccentricity = np.array(eccentricity) + threshold
    eccentricity[final_mask_L_dorsal != 1] = 0

    # Binarizing
    if binarize == True:
        eccentricity[(eccentricity >= 0) & (eccentricity <= 2)] = 0 + threshold
        eccentricity[(eccentricity > 2) & (eccentricity <= 4)] = 2 + threshold
        eccentricity[(eccentricity > 4) & (eccentricity <= 6)] = 4 + threshold
        eccentricity[(eccentricity > 6)] = 6 + threshold
        eccentricity[final_mask_L_dorsal != 1] = 0

    # Plotting
    view = plotting.view_surf(
        surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '../data'
                           '/S1200_7T_Retinotopy181.L.sphere.32k_fs_LR.surf.gii'),
        surf_map=np.reshape(eccentricity[0:32492], (-1)), bg_map=background,
        cmap='gist_rainbow_r', black_bg=False, symmetric_cmap=False,
        threshold=threshold, vmax=8)
    return view.open_in_browser()


if __name__ == '__main__':
    curv_background_subject = '111312'  # background only

    for i in range(6):
        eccentricity_plot(curv_background_subject, './../data/', cluster=i)
