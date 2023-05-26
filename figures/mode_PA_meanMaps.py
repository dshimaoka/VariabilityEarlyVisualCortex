import numpy as np
import scipy.io
import os.path as osp
import sys
import argparse

sys.path.append('..')

from nilearn import plotting
from functions.def_ROIs_EarlyVisualAreas import roi as ROI
from functions.def_ROIs_DorsalEarlyVisualCortex import roi

def topographic_map_plot(subject_id, path, modality, hemisphere, cluster):
    """
    Plot the mean topographic_map of the early visual cortex.
    Parameters
    ----------
    subject_id : int
        Subject ID for background curvature.
    path : str  
        Path to the data.
    modality : str
        Modality.
    hemisphere : str
        Hemisphere.
    cluster : str
        cluster index.

    Returns
    -------
    Plot of the mean topographic_map from the given cluster.
    """

    # Loading the curvature topographic_map
    curv = scipy.io.loadmat(osp.join(path, 'cifti_curvature_all.mat'))[
        'cifti_curvature']
    background = np.reshape(
        curv['x' + subject_id + '_curvature'][0][0][0:32492], (-1))

    # Background settings
    threshold = 1  # threshold for the curvature topographic_map
    nocurv = np.isnan(background)
    background[nocurv == 1] = 0
    background[background < 0] = 0
    background[background > 0] = 1

    # Mask - Early visual cortex
    final_mask_L_ROI, final_mask_R_ROI, index_L_mask_ROI, index_R_mask_ROI = \
        ROI(['ROI'])

    # Number of nodes
    number_cortical_nodes = int(64984)
    number_hemi_nodes = int(number_cortical_nodes / 2)

    # Loading the data
    topographic_map = np.zeros((32492, 1))
    data = np.load('./../output/mean' + modality + '_PAclustering_cluster' +
                   str(cluster) + '_' + hemisphere + '.npz')['list']
    if hemisphere == 'LH':
        topographic_map[final_mask_L_ROI == 1] = np.reshape(
            data, (-1, 1))

        # Masking and Reshifting
        topographic_map = np.array(topographic_map)
        if modality == 'polarAngle':
            minus = topographic_map > 180
            sum = topographic_map < 180
            topographic_map[minus] = topographic_map[minus] - 180 + threshold
            topographic_map[sum] = topographic_map[sum] + 180 + threshold
        elif modality == 'eccentricity':
            topographic_map[final_mask_L_ROI ==
                            1] = topographic_map[final_mask_L_ROI == 1] + threshold
        elif modality == 'meanbold':
            topographic_map[final_mask_L_ROI ==
                            1] = topographic_map[final_mask_L_ROI == 1] + threshold
            # print(np.min(topographic_map[final_mask_L_ROI == 1]))
        topographic_map[final_mask_L_ROI != 1] = 0

        # Plotting
        cmap = 'gist_rainbow_r'
        if modality == 'polarAngle':
            vmax = 360 + threshold
        elif modality == 'eccentricity':
            vmax = 8 + threshold
        elif modality == 'meanbold':
            vmax = 1.8
            cmap = 'gist_heat'

        view = plotting.view_surf(
            surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '../data'
                               '/S1200_7T_Retinotopy181.L.sphere.32k_fs_LR.surf.gii'),
            surf_map=np.reshape(topographic_map[0:32492], (-1)), bg_map=background,
            cmap=cmap, black_bg=False, symmetric_cmap=False,
            threshold=threshold, vmax=vmax)
        return view.open_in_browser()
    else:
        topographic_map[final_mask_R_ROI == 1] = np.reshape(
            data, (-1, 1))

        # Masking and shifting
        topographic_map = np.array(topographic_map)
        if modality == 'polarAngle':
            minus = topographic_map > 180
            sum = topographic_map < 180
            topographic_map[minus] = topographic_map[minus] - 180 + threshold
            topographic_map[sum] = topographic_map[sum] + 180 + threshold
        elif modality == 'eccentricity':
            topographic_map[final_mask_L_ROI ==
                            1] = topographic_map[final_mask_L_ROI == 1] + threshold
        elif modality == 'meanbold':
            topographic_map[final_mask_L_ROI ==
                            1] = topographic_map[final_mask_L_ROI == 1] + threshold
        topographic_map[final_mask_L_ROI != 1] = 0
        topographic_map[final_mask_R_ROI != 1] = 0

        # Plotting
        cmap = 'gist_rainbow'
        if modality == 'polarAngle':
            vmax = 360 + threshold
        elif modality == 'eccentricity':
            vmax = 8 + threshold
            cmap = 'gist_rainbow_r'
        elif modality == 'meanbold':
            vmax = 1.8
            cmap = 'gist_heat'

        view = plotting.view_surf(
            surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '../data'
                               '/S1200_7T_Retinotopy181.R.sphere.32k_fs_LR.surf.gii'),
            surf_map=np.reshape(topographic_map[0:32492], (-1)), bg_map=background,
            cmap=cmap, black_bg=False, symmetric_cmap=False,
            threshold=threshold, vmax=vmax)
        return view.open_in_browser()


if __name__ == '__main__':
    curv_background_subject = '111312' # background only
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--hemisphere', type=str, default='LH')
    parser.add_argument('--modality', type=str, default='polarAngle')
    args = parser.parse_args()

    for i in range(1, 7):
        topographic_map_plot(curv_background_subject,
                             './../data/', args.modality, args.hemisphere, cluster=i)
