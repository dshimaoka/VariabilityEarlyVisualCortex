import numpy as np
import scipy.io
import os.path as osp
import sys

sys.path.append('..')

from nilearn import plotting
from functions.def_ROIs_DorsalEarlyVisualCortex import roi as ROI
from functions.def_ROIs_EarlyVisualAreas import roi

def polarAngle_plot(subject_id, path, dorsal_only=False, binarize=False, save=False, save_path=None):
    """
    Plot the polar angle map of the early visual cortex.
    Parameters
    ----------
    subject_id : int
        Subject ID.
    path : str  
        Path to the data.
    dorsal_only : bool, optional
        Plot only the dorsal portion of the early visual cortex. The default is False.
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

    # Background settings
    threshold = 1  # threshold for the curvature map
    nocurv = np.isnan(background)
    background[nocurv == 1] = 0
    background[background < 0] = 0
    background[background > 0] = 1

    # Early visual cortex
    label_primary_visual_areas = ['ROI']
    final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
        label_primary_visual_areas)

    # Dorsal portion
    label_primary_visual_areas = ['ROI']
    final_mask_L_dorsal, final_mask_R_dorsal, index_L_mask_dorsal, index_R_mask_dorsal = ROI(
        label_primary_visual_areas)

    # Number of nodes
    number_cortical_nodes = int(64984)
    number_hemi_nodes = int(number_cortical_nodes / 2)

    # Loading the polarAngleictions
    polarAngle = np.zeros((32492, 1))
    data = scipy.io.loadmat(osp.join(path, 'cifti_polarAngle_all.mat'))[
        'cifti_polarAngle']
    if dorsal_only == False:
        polarAngle[final_mask_L == 1] = np.reshape(
            data['x' + str(subject_id) + '_fit1_polarangle_msmall'][0][0][
                0:number_hemi_nodes].reshape(
                (number_hemi_nodes))[final_mask_L == 1], (-1, 1))
    else:
        polarAngle[final_mask_L*final_mask_L_dorsal == 1] = np.reshape(
            data['x' + str(subject_id) + '_fit1_polarangle_msmall'][0][0][
                0:number_hemi_nodes].reshape(
                (number_hemi_nodes))[final_mask_L*final_mask_L_dorsal == 1], (-1, 1))

    # Masking
    polarAngle = np.array(polarAngle) + threshold
    if dorsal_only == False:
        polarAngle[final_mask_L != 1] = 0
    else:
        polarAngle[final_mask_L*final_mask_L_dorsal != 1] = 0

    # Binarizing values
    if binarize == True:
        polarAngle[(polarAngle >= 0) & (polarAngle <= 45)] = 0 + threshold
        polarAngle[(polarAngle > 45) & (polarAngle <= 180)] = 90 + threshold
        polarAngle[(polarAngle >= 315) & (polarAngle <= 360)] = 360 + threshold
        polarAngle[(polarAngle > 180) & (polarAngle < 315)] = 270 + threshold
        if dorsal_only == False:
            polarAngle[final_mask_L != 1] = 0
        else:
            polarAngle[final_mask_L*final_mask_L_dorsal != 1] = 0

    # Plotting
    view = plotting.view_surf(
        surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '../data'
                           '/S1200_7T_Retinotopy181.L.sphere.32k_fs_LR.surf.gii'),
        surf_map=np.reshape(polarAngle[0:32492], (-1)), bg_map=background,
        cmap='gist_rainbow_r', black_bg=False, symmetric_cmap=False,
        threshold=threshold, vmax=361)
    # view.open_in_browser()

    if save == True:
        view.save_as_html(
            osp.join(save_path, 'polarAngle_dorsal_' + subject_id + '.html'))
    return view


if __name__ == '__main__':
    # subjects =['585256', '146735', '157336', '114823', '581450', '725751'] # figure 1
    # subjects = ['108323', '105923', '178243', '193845', '157336', '169343'] # figure 4 - similar
    # subjects = ['148133', '176542', '181232', '132118', '177140', '204521'] # figure 4 - dissimilar
    # subjects =['573249', '130114', '171633', '249947', '108323', '164131', '552241', '966975', '525541'] # Supplementary Figure 3
    # subjects = ['105923', '109123', '111514', '114823', '116726', '125525',
    #     '167036', '169747', '176542', '198653', '205220', '360030',
    #     '385046', '429040', '581450', '671855', '706040', '770352',
    #     '771354', '789373', '814649', '825048', '826353', '859671',
    #     '942658', '973770'] # Supplementary Figure 4
    # subjects = ['725751', '118225', '178243', '192641', '115825', '169040', '644246', '995174', '320826'] # Supplementary Figure 5
    # subjects = ['818859', '131217', '182436', '263436', '126426', '178142', '757764', '541943', '330324'] # Supplementary Figure 6
    # subjects = ['905147', '135124', '195041', '397760', '130518', '177140', '572045', '401422', '111312'] # Supplementary Figure 7
    subjects = ['536647', '169343', '214019', '251833', '156334',
                '204521', '389357', '878776', '203418']  # Supplementary Figure 8
    for subject in subjects:
        path = './../data'
        polarAngle_plot(subject, path, binarize=True,
                        dorsal_only=True).open_in_browser()
