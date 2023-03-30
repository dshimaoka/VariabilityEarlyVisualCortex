import scipy
import os.path as osp
import os
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from functions.def_ROIs_WangParcels import roi
from functions.def_ROIs_EarlyVisualAreas import roi as ROI
from functions.error_metrics import smallest_angle
from functions.individual_variability import grab_data


def BOLDamplitude(list_of_ind, list_of_areas):
    """Load and return individual's data.

    Args:
        modality (string): modality of the data ('polarAngle', 'eccentricity',
            'myelin', 'curvature')
        subject_ID (string): HCP ID
        roi_mask (numpy array): Mask of the region of interest from the left
            (or the right) hemisphere (32492,)
        hemisphere (string): 'LH' for left or 'RH' for right

    Returns:
        data_ind (numpy array): Individual's data from the region of interest
            (number_of_nodes,)
    """

    # Mask for the early visual cortex - ROI
    final_mask_L_ROI, final_mask_R_ROI, index_L_mask_ROI, index_R_mask_ROI = \
        ROI(['ROI'])

    # Data
    data = {'gain_data_LH': [],
            'gain_data_RH': [],
            'meanSignal_data_LH': [],
            'meanSignal_data_RH': []}
    for i in range(len(list_of_ind)):
        data['gain_data_LH'].append(
            grab_data('gain', str(list_of_ind[i]), final_mask_L_ROI,
                      'LH'))
        data['gain_data_RH'].append(
            grab_data('gain', str(list_of_ind[i]), final_mask_R_ROI,
                      'RH'))
        data['meanSignal_data_LH'].append(
            grab_data('meanbold', str(list_of_ind[i]), final_mask_L_ROI,
                      'LH'))
        data['meanSignal_data_RH'].append(
            grab_data('meanbold', str(list_of_ind[i]), final_mask_R_ROI,
                      'RH'))


    # Mask for specific visual areas
    final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
        list_of_areas)

    # Selecting only vertices of final_mask_L within the visual cortex (ROI)
    final_mask_L = final_mask_L[final_mask_L_ROI == 1]
    final_mask_R = final_mask_R[final_mask_R_ROI == 1]

    # Mask for eccentricity range
    eccentricity_mask_LH = np.reshape(
        np.load('./../main/MaskEccentricity_'
                'above1below8ecc_LH.npz')['list'], (-1))
    eccentricity_mask_RH = np.reshape(
        np.load('./../main/MaskEccentricity_'
                'above1below8ecc_RH.npz')['list'], (-1))

    # Final mask - values equal to 2 mean nodes with visual area and within
    # eccentricity range
    mask_LH = final_mask_L + eccentricity_mask_LH
    mask_RH = final_mask_R + eccentricity_mask_RH

    # Difference of the individuals' curvature maps to the mean
    data['PercentSignalChange_LH'] = []
    data['PercentSignalChange_RH'] = []

    for i in range(len(data['gain_data_LH'])):
        data['PercentSignalChange_LH'].append(
            np.reshape((data['gain_data_LH'][i] * 100 /data['meanSignal_data_LH'][i])[
                            mask_LH == 2], (-1)))
        data['PercentSignalChange_RH'].append(
            np.reshape((data['gain_data_RH'][i] * 100 /data['meanSignal_data_RH'][i])[
                            mask_RH == 2], (-1)))
    print(np.shape(data['PercentSignalChange_LH']))
    mean_LH = np.mean(data['PercentSignalChange_LH'], axis=1)
    mean_RH = np.mean(data['PercentSignalChange_RH'], axis=1)
    return mean_LH, mean_RH