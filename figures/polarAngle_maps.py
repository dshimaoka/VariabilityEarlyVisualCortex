import numpy as np
import scipy.io
import os.path as osp
import sys

sys.path.append('..')

from functions.def_ROIs_EarlyVisualAreas import roi
from functions.def_ROIs_DorsalEarlyVisualCortex import roi as ROI
from nilearn import plotting

def polarAngle_dorsal_plot(subject_id, path, binarize = False, save = False, save_path = None):
    """
    Plot the polar angle map for the dorsal portion of the early visual cortex.
    Parameters
    ----------
    subject_id : int
        Subject ID.
    path : str  
        Path to the data.
    binarize : bool, optional
        Binarize the polar angle map. The default is False.
    save : bool, optional
        Save the figure. The default is False.
    save_path : str, optional
        Path to save the figure. The default is None.
    Returns
    ------- 
    view : nilearn.plotting.view_img
    """
    # Loading the curvature map
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

    # ROI settings
    label_primary_visual_areas = ['ROI']
    final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
        label_primary_visual_areas)

    # Dorsal portion
    label_primary_visual_areas = ['ROI']
    final_mask_L_dorsal, final_mask_R_dorsal, index_L_mask_dorsal, index_R_mask_dorsal = ROI(
        label_primary_visual_areas)


    polarAngle = np.zeros((32492, 1))

    number_cortical_nodes = int(64984)
    number_hemi_nodes = int(number_cortical_nodes / 2)

    # Loading the predictions
    data = scipy.io.loadmat(osp.join(path, 'cifti_polarAngle_all.mat'))[
        'cifti_polarAngle']

    polarAngle[final_mask_L*final_mask_L_dorsal == 1] = np.reshape(
        data['x' + str(subject_id) + '_fit1_polarangle_msmall'][0][0][
        0:number_hemi_nodes].reshape(
            (number_hemi_nodes))[final_mask_L*final_mask_L_dorsal == 1], (-1, 1))

    # Masking
    pred = np.array(polarAngle) + threshold
    pred[final_mask_L*final_mask_L_dorsal != 1] = 0

    # Binarizing values
    if binarize==True:
        pred[(pred >= 0) & (pred <= 45)] = 0 + threshold
        pred[(pred > 45) & (pred <= 180)]= 90 + threshold
        pred[(pred >= 315) & (pred <= 360)] = 360 + threshold
        pred[(pred > 180) & (pred < 315)] = 270 + threshold
        pred[final_mask_L*final_mask_L_dorsal != 1] = 0

    # Empirical map
    view = plotting.view_surf(
        surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '../data'
                '/S1200_7T_Retinotopy181.L.sphere.32k_fs_LR.surf.gii'),
        surf_map=np.reshape(pred[0:32492], (-1)), bg_map=background,
        cmap='gist_rainbow_r', black_bg=False, symmetric_cmap=False,
        threshold=threshold, vmax=361)
    # view.open_in_browser()
    if save == True:
        view.save_as_html(osp.join(save_path,'polarAngle_dorsal_' + subject_id + '.html'))
    return view

if __name__ == '__main__':
    subjects =['585256', '146735', '157336', '114823', '581450', '725751'] # figure 1
    # subjects = ['108323', '105923', '178243', '193845', '157336', '169343'] # figure 4 - similar
    # subjects = ['148133', '176542', '181232', '132118', '177140', '204521'] # figure 4 - dissimilar
    # subjects = ['105923', '109123', '111514', '114823', '116726', '125525',
    #     '167036', '169747', '176542', '198653', '205220', '360030',
    #     '385046', '429040', '581450', '671855', '706040', '770352',
    #     '771354', '789373', '814649', '825048', '826353', '859671',
    #     '942658', '973770'] # Supplementary Figure 2


    for subject in subjects:
        path = './../data'
        polarAngle_dorsal_plot(subject, path).open_in_browser()
