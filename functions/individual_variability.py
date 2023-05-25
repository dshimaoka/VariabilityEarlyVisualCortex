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


def grab_data(modality, subject_ID, roi_mask, hemisphere):
    """Load and return individual's data.

    Args:
        modality (string): modality of the data ('polarAngle', 'eccentricity',
            'meanbold', 'curvature')
        subject_ID (string): HCP ID
        roi_mask (numpy array): Mask of the region of interest from the left
            (or the right) hemisphere (32492,)
        hemisphere (string): 'LH' for left or 'RH' for right

    Returns:
        data_ind (numpy array): Individual's data from the region of interest
            (number_of_nodes,)
    """
    path = './../data'
    # Number of nodes
    number_cortical_nodes = int(64984)
    number_hemi_nodes = int(number_cortical_nodes / 2)

    data = \
        scipy.io.loadmat(
            osp.join(path, 'cifti_' + str(modality) + '_all.mat'))[
            'cifti_' + str(modality)]

    if hemisphere == 'LH':
        # anatomical maps
        if modality == 'curvature':
            data_ind = np.reshape(
                data['x' + str(subject_ID) + '_curvature'][0][0][
                0:number_hemi_nodes].reshape(
                    (number_hemi_nodes))[roi_mask == 1], (-1, 1))
        if modality == 'myelin':
            data_ind = np.reshape(
                data['x' + str(subject_ID) + '_myelinmap'][0][0][
                0:number_hemi_nodes].reshape(
                    (number_hemi_nodes))[roi_mask == 1], (-1, 1))

        # functional maps
        if modality == 'gain':
            data_ind = np.reshape(
                data['x' + str(subject_ID) + '_fit1_gain_msmall'][0][
                    0][
                0:number_hemi_nodes].reshape(
                    (number_hemi_nodes))[roi_mask == 1], (-1, 1)) 
        if modality == 'meanbold':
            data_ind_tmp = data['x' + str(subject_ID) + '_fit1_meanbold_msmall'][0][
                    0][0:number_hemi_nodes].reshape((number_hemi_nodes))
            data_ind_tmp[np.isnan(data_ind_tmp) == 1] = 0
            # Normalisation by dividing the value of each voxel by the maximum intensity
            data_ind_tmp = data_ind_tmp / np.max(data_ind_tmp)
            data_ind = np.reshape(data_ind_tmp[roi_mask == 1], (-1, 1)) 
        if modality == 'eccentricity':
            data_ind = np.reshape(
                data['x' + str(subject_ID) + '_fit1_eccentricity_msmall'][0][
                    0][
                0:number_hemi_nodes].reshape(
                    (number_hemi_nodes))[roi_mask == 1], (-1, 1))
        if modality == 'polarAngle':
            data_ind = np.reshape(
                data['x' + str(subject_ID) + '_fit1_polarangle_msmall'][0][0][
                0:number_hemi_nodes].reshape(
                    (number_hemi_nodes))[roi_mask == 1], (-1, 1))

            # Shifting PA values
            sum = data_ind < 180
            minus = data_ind > 180
            data_ind[sum] = data_ind[sum] + 180
            data_ind[minus] = data_ind[minus] - 180

    if hemisphere == 'RH':
        # anatomical maps
        if modality == 'curvature':
            data_ind = np.reshape(
                data['x' + str(subject_ID) + '_curvature'][0][0][
                number_hemi_nodes:number_cortical_nodes].reshape(
                    (number_hemi_nodes))[roi_mask == 1], (-1, 1))
        if modality == 'myelin':
            data_ind = np.reshape(
                data['x' + str(subject_ID) + '_myelinmap'][0][0][
                number_hemi_nodes:number_cortical_nodes].reshape(
                    (number_hemi_nodes))[roi_mask == 1], (-1, 1))
            
        # functional maps
        if modality == 'gain':
            data_ind = np.reshape(
                data['x' + str(subject_ID) + '_fit1_gain_msmall'][0][
                    0][
                number_hemi_nodes:number_cortical_nodes].reshape(
                    (number_hemi_nodes))[roi_mask == 1], (-1, 1)) 
        if modality == 'meanbold':
            data_ind_tmp = data['x' + str(subject_ID) + '_fit1_meanbold_msmall'][0][
                    0][number_hemi_nodes:number_cortical_nodes].reshape((number_hemi_nodes))
            data_ind_tmp[np.isnan(data_ind_tmp) == 1] = 0
            # Normalisation by dividing the value of each voxel by the maximum intensity
            data_ind_tmp = data_ind_tmp / np.max(data_ind_tmp)
            data_ind = np.reshape(data_ind_tmp[roi_mask == 1], (-1, 1)) 
        if modality == 'eccentricity':
            data_ind = np.reshape(
                data['x' + str(subject_ID) + '_fit1_eccentricity_msmall'][0][
                    0][
                number_hemi_nodes:number_cortical_nodes].reshape(
                    (number_hemi_nodes))[roi_mask == 1], (-1, 1))
        if modality == 'polarAngle':
            data_ind = np.reshape(
                data['x' + str(subject_ID) + '_fit1_polarangle_msmall'][0][0][
                number_hemi_nodes:number_cortical_nodes].reshape(
                    (number_hemi_nodes))[roi_mask == 1], (-1, 1))

    data_ind[np.isnan(data_ind) == 1] = 0
    return data_ind


def grab_data_fit(modality, subject_ID, roi_mask, hemisphere, fit_name):
    """Load and return individual's data given a specific pRF mapping fit.

    Args:
        modality (string): modality of the data ('polarAngle', 'eccentricity')
        subject_ID (string): HCP ID
        roi_mask (numpy array): Mask of the region of interest from the left
            (or the right) hemisphere (32492,)
        hemisphere (string): 'LH' for left or 'RH' for right
        fit_name (string): 'fit2' or 'fit3' for independent fits from Benson
        et al. (2018)

    Returns:
        data_ind (numpy array): Individual's data from the region of interest
            (number_of_nodes,)
    """
    path = './../data'

    # Number of nodes
    number_cortical_nodes = int(64984)
    number_hemi_nodes = int(number_cortical_nodes / 2)

    data = \
        scipy.io.loadmat(
            osp.join(path,
                     'cifti_' + str(modality) + '_' + fit_name + '_all.mat'))[
            'cifti_' + str(modality)]

    if hemisphere == 'LH':
        if modality == 'eccentricity':
            data_ind = np.reshape(
                data['x' + str(
                    subject_ID) + '_' + fit_name + '_eccentricity_msmall'][0][
                    0][
                0:number_hemi_nodes].reshape(
                    (number_hemi_nodes))[roi_mask == 1], (-1, 1))
        if modality == 'polarAngle':
            data_ind = np.reshape(
                data['x' + str(
                    subject_ID) + '_' + fit_name + '_polarangle_msmall'][0][0][
                0:number_hemi_nodes].reshape(
                    (number_hemi_nodes))[roi_mask == 1], (-1, 1))

            # Shifting PA values
            sum = data_ind < 180
            minus = data_ind > 180
            data_ind[sum] = data_ind[sum] + 180
            data_ind[minus] = data_ind[minus] - 180

    if hemisphere == 'RH':
        if modality == 'eccentricity':
            data_ind = np.reshape(
                data['x' + str(
                    subject_ID) + '_' + fit_name + '_eccentricity_msmall'][0][
                    0][
                number_hemi_nodes:number_cortical_nodes].reshape(
                    (number_hemi_nodes))[roi_mask == 1], (-1, 1))
        if modality == 'polarAngle':
            data_ind = np.reshape(
                data['x' + str(
                    subject_ID) + '_' + fit_name + '_polarangle_msmall'][0][0][
                number_hemi_nodes:number_cortical_nodes].reshape(
                    (number_hemi_nodes))[roi_mask == 1], (-1, 1))

    data_ind[np.isnan(data_ind) == 1] = 0
    return data_ind


def difference_score(modality, list_of_ind, list_of_areas):
    """Determine the vertex-wise difference between an individual's data and the
    average map.

    Args:
        modality (string): modality of the data ('polarAngle', 'eccentricity',
            'meanbold', 'curvature')
        list_of_ind (list): list with HCP IDs
        list_of_areas (list): list with the names of visual areas

    Returns:
        mean_LH (numpy array), mean_RH(numpy array): Mean difference across
            vertices per individual for corresponding visual areas in both left
            and right hemispheres
    """

    # Mask for the early visual cortex - ROI
    final_mask_L_ROI, final_mask_R_ROI, index_L_mask_ROI, index_R_mask_ROI = \
        ROI(['ROI'])

    # Data
    data = {str(modality) + '_data_LH': [],
            str(modality) + '_data_RH': []}
    for i in range(len(list_of_ind)):
        data[str(modality) + '_data_LH'].append(
            grab_data(str(modality), str(list_of_ind[i]), final_mask_L_ROI,
                      'LH'))
        data[str(modality) + '_data_RH'].append(
            grab_data(str(modality), str(list_of_ind[i]), final_mask_R_ROI,
                      'RH'))

    # Mean maps
    data['mean_' + str(modality) + '_LH'] = np.mean(
        data[str(modality) + '_data_LH'], axis=0)
    data['mean_' + str(modality) + '_RH'] = np.mean(
        data[str(modality) + '_data_RH'], axis=0)

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
    data['difference_' + str(modality) + '_LH'] = []
    data['difference_' + str(modality) + '_RH'] = []
    if modality == 'eccentricity' or modality == 'polarAngle':
        for i in range(len(data[str(modality) + '_data_LH'])):
            data['difference_' + str(modality) + '_LH'].append(smallest_angle(
                data['mean_' + str(modality) + '_LH'] * np.pi / 180,
                data[str(modality) + '_data_LH'][i] * np.pi / 180)[
                                                                   mask_LH
                                                                   == 2])
            data['difference_' + str(modality) + '_RH'].append(smallest_angle(
                data['mean_' + str(modality) + '_RH'] * np.pi / 180,
                data[str(modality) + '_data_RH'][i] * np.pi / 180)[
                                                                   mask_RH
                                                                   == 2])
    else:
        for i in range(len(data[str(modality) + '_data_LH'])):
            data['difference_' + str(modality) + '_LH'].append(
                np.absolute((data['mean_' + str(modality) + '_LH'] -
                             data[str(modality) + '_data_LH'][i])[
                                mask_LH == 2]))

            data['difference_' + str(modality) + '_RH'].append(
                np.absolute((data['mean_' + str(modality) + '_RH'] -
                             data[str(modality) + '_data_RH'][i])[
                                mask_RH == 2]))

    mean_LH = np.mean(data['difference_' + str(modality) + '_LH'], axis=1)
    mean_RH = np.mean(data['difference_' + str(modality) + '_RH'], axis=1)
    return mean_LH, mean_RH


def intra_individual_difference_score(modality, list_of_ind, list_of_areas):
    """Determine the vertex-wise difference between different pRF mapping fits
     for each individual.

    Args:
        modality (string): modality of the data ('polarAngle', 'eccentricity')
        list_of_ind (list): list with HCP IDs
        list_of_areas (list): list with the names of visual areas

    Returns:
        mean_LH (numpy array), mean_RH(numpy array): Mean difference across
            vertices per individual for corresponding visual areas in both left
            and right hemispheres
    """

    # Mask for the early visual cortex - ROI
    final_mask_L_ROI, final_mask_R_ROI, index_L_mask_ROI, index_R_mask_ROI = \
        ROI(['ROI'])

    # Data
    data_fit2 = {str(modality) + '_data_LH': [],
                 str(modality) + '_data_RH': []}
    data_fit3 = {str(modality) + '_data_LH': [],
                 str(modality) + '_data_RH': []}

    for i in range(len(list_of_ind)):
        data_fit2[str(modality) + '_data_LH'].append(
            grab_data_fit(str(modality), str(list_of_ind[i]), final_mask_L_ROI,
                          'LH', 'fit2'))
        data_fit3[str(modality) + '_data_LH'].append(
            grab_data_fit(str(modality), str(list_of_ind[i]), final_mask_L_ROI,
                          'LH', 'fit3'))

        data_fit2[str(modality) + '_data_RH'].append(
            grab_data_fit(str(modality), str(list_of_ind[i]), final_mask_R_ROI,
                          'RH', 'fit2'))
        data_fit3[str(modality) + '_data_RH'].append(
            grab_data_fit(str(modality), str(list_of_ind[i]), final_mask_R_ROI,
                          'RH', 'fit3'))

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
    data = {}
    data['difference_' + str(modality) + '_LH'] = []
    data['difference_' + str(modality) + '_RH'] = []

    for i in range(len(data_fit2[str(modality) + '_data_LH'])):
        data['difference_' + str(modality) + '_LH'].append(smallest_angle(
            data_fit2[str(modality) + '_data_LH'][i] * np.pi / 180,
            data_fit3[str(modality) + '_data_LH'][i] * np.pi / 180)[
                                                               mask_LH
                                                               == 2])
        data['difference_' + str(modality) + '_RH'].append(smallest_angle(
            data_fit2[str(modality) + '_data_RH'][i] * np.pi / 180,
            data_fit3[str(modality) + '_data_RH'][i] * np.pi / 180)[
                                                               mask_RH
                                                               == 2])

    mean_LH = np.mean(data['difference_' + str(modality) + '_LH'], axis=1)
    mean_RH = np.mean(data['difference_' + str(modality) + '_RH'], axis=1)
    return mean_LH, mean_RH


def difference_plots(modality, differences_dorsal_final,
                     differences_ventral_final):
    """Generate left versus right hemispheres plots of individual variability
        in topographic features.

    Args:
        modality (string): modality of the data ('polarAngle', 'eccentricity',
            'meanbold', 'curvature')
        differences_dorsal_final (numpy array): Mean difference across
            vertices per individual for corresponding visual areas in the left
            hemisphere (output from difference_score function)
        differences_ventral_final (numpy array): Mean difference across
            vertices per individual for corresponding visual areas in the right
            hemisphere (output from difference_score function)

    Returns:
        df (pandas DataFrame): summary data used to generate the plots
    """

    fig = plt.figure(figsize=(10, 5))

    df_1 = pd.DataFrame(
        columns=['HCP_ID', 'Mean difference from the average map',
                 'Hemisphere',
                 'Hemi visual area', 'Visual area', 'Half'],
        data=differences_dorsal_final.T)
    df_1['Mean difference from the average map'] = df_1[
        'Mean difference from the average map'].astype(float)

    fig.add_subplot(1, 2, 1)
    palette_1 = [sns.color_palette("PRGn_r")[5],
                 sns.color_palette("PRGn_r")[0]]
    ax = sns.violinplot(x="Hemi visual area",
                        y="Mean difference from the average map",
                        hue="Hemisphere", data=df_1,
                        palette=palette_1, split=True, inner="quartile")
    if modality == 'polarAngle':
        plt.ylim([0, 90])
        plt.title('Polar angle')
    if modality == 'eccentricity':
        plt.ylim([0, 3])
        plt.title('Eccentricity')
    if modality == 'meanbold':
        plt.ylim([0, .3])
        plt.title('Mean BOLD')
    if modality == 'curvature':
        plt.ylim([0, .3])
        plt.title('Curvature')
    sns.despine()

    df_2 = pd.DataFrame(
        columns=['HCP_ID', 'Mean difference from the average map',
                 'Hemisphere',
                 'Hemi visual area', 'Visual area', 'Half'],
        data=differences_ventral_final.T)
    df_2['Mean difference from the average map'] = df_2[
        'Mean difference from the average map'].astype(float)

    fig.add_subplot(1, 2, 2)
    palette_2 = [sns.color_palette("PRGn_r")[4],
                 sns.color_palette("PRGn_r")[1]]
    ax = sns.violinplot(x="Hemi visual area",
                        y="Mean difference from the average map",
                        hue="Hemisphere", data=df_2,
                        palette=palette_2, split=True, inner="quartile")
    if modality == 'polarAngle':
        plt.ylim([0, 90])
        plt.title('Polar angle')
    if modality == 'eccentricity':
        plt.ylim([0, 3])
        plt.title('Eccentricity')
    if modality == 'meanbold':
        plt.ylim([0, .3])
        plt.title('Mean BOLD')
    if modality == 'curvature':
        plt.ylim([0, .3])
        plt.title('Curvature')
    sns.despine()

    # Create an output folder if it doesn't already exist
    directory = './../figures/other'
    if not osp.exists(directory):
        os.makedirs(directory)

    plt.savefig('./../figures/other/MeanDifFromTheMean_combined_' + str(modality) +
                '_181participants.pdf', format="pdf")
    plt.show()
    df = pd.concat([df_1, df_2])
    return df


def intraIndividual_difference_plots(modality, differences_dorsal_final,
                                     differences_ventral_final):
    """Generate left versus right hemispheres plots of intra individual variability
        in topographic features.

    Args:
        modality (string): modality of the data ('polarAngle', 'eccentricity')
        differences_dorsal_final (numpy array): Mean difference across
            vertices per individual for corresponding visual areas in the left
            hemisphere (output from difference_score function)
        differences_ventral_final (numpy array): Mean difference across
            vertices per individual for corresponding visual areas in the right
            hemisphere (output from difference_score function)

    Returns:
        df (pandas DataFrame): summary data used to generate the plots
    """

    fig = plt.figure(figsize=(10, 5))

    df_1 = pd.DataFrame(
        columns=['HCP_ID',
                 "Mean intra-individual variability in pRF estimates",
                 'Hemisphere',
                 'Hemi visual area', 'Visual area', 'Half'],
        data=differences_dorsal_final.T)
    df_1["Mean intra-individual variability in pRF estimates"] = df_1[
        "Mean intra-individual variability in pRF estimates"].astype(float)

    fig.add_subplot(1, 2, 1)
    palette_1 = [sns.color_palette("PRGn_r")[5],
                 sns.color_palette("PRGn_r")[0]]
    ax = sns.violinplot(x="Hemi visual area",
                        y="Mean intra-individual variability in pRF estimates",
                        hue="Hemisphere", data=df_1,
                        palette=palette_1, split=True, inner="quartile")
    if modality == 'polarAngle':
        plt.ylim([0, 90])
        plt.title('Polar angle')
    if modality == 'eccentricity':
        plt.ylim([0, 3])
        plt.title('Eccentricity')
    sns.despine()

    df_2 = pd.DataFrame(
        columns=['HCP_ID',
                 "Mean intra-individual variability in pRF estimates",
                 'Hemisphere',
                 'Hemi visual area', 'Visual area', 'Half'],
        data=differences_ventral_final.T)
    df_2["Mean intra-individual variability in pRF estimates"] = df_2[
        "Mean intra-individual variability in pRF estimates"].astype(float)

    fig.add_subplot(1, 2, 2)
    palette_2 = [sns.color_palette("PRGn_r")[4],
                 sns.color_palette("PRGn_r")[1]]
    ax = sns.violinplot(x="Hemi visual area",
                        y="Mean intra-individual variability in pRF estimates",
                        hue="Hemisphere", data=df_2,
                        palette=palette_2, split=True, inner="quartile")
    if modality == 'polarAngle':
        plt.ylim([0, 90])
        plt.title('Polar angle')
    if modality == 'eccentricity':
        plt.ylim([0, 3])
        plt.title('Eccentricity')
    sns.despine()

    # Create an output folder if it doesn't already exist
    directory = './../figures/other'
    if not osp.exists(directory):
        os.makedirs(directory)

    plt.savefig('./../figures/other/IntraIndividual_MeanDifFit2vsFit3_combined_' + str(modality) + '_181participants.pdf', format="pdf")
    plt.show()
    df = pd.concat([df_1, df_2])
    return df


def difference_plots_sameHemi(data, modality):
    """Generate dorsal versus ventral plots of individual variability
        of topographic features.

    Args:
        data (pandas DataFrame):  summary data used to generate the plots,
            which is the output of difference_plots
        modality (string): modality of the data ('polarAngle', 'eccentricity',
            'meanbold', 'curvature')
    Returns:
        plt.show(): plot of the data

    """
    fig = plt.figure(figsize=(10, 5))
    fig.add_subplot(1, 2, 1)
    palette_2 = [sns.color_palette("PRGn_r")[5],
                 sns.color_palette("PRGn_r")[4]]
    sns.violinplot(x="Visual area", y="Mean difference from the average map",
                   hue="Half", data=data[data['Hemisphere'] == 'LH'],
                   palette=palette_2, split=True, inner="quartile")
    if modality == 'polarAngle':
        plt.ylim([0, 90])
        plt.title('Polar angle' + ' - ' + str('LH'))
    if modality == 'eccentricity':
        plt.ylim([0, 3])
        plt.title('Eccentricity' + ' - ' + str('LH'))
    if modality == 'meanbold':
        plt.ylim([0, .3])
        plt.title('Mean BOLD' + ' - ' + str('LH'))
    if modality == 'curvature':
        plt.ylim([0, .3])
        plt.title('Curvature' + ' - ' + str('LH'))
    sns.despine()

    fig.add_subplot(1, 2, 2)
    palette_1 = [sns.color_palette("PRGn_r")[0],
                 sns.color_palette("PRGn_r")[1]]

    sns.violinplot(x="Visual area", y="Mean difference from the average map",
                   hue="Half", data=data[data['Hemisphere'] == 'RH'],
                   palette=palette_1, split=True, inner="quartile")
    if modality == 'polarAngle':
        plt.ylim([0, 90])
        plt.title('Polar angle' + ' - ' + str('RH'))
    if modality == 'eccentricity':
        plt.ylim([0, 3])
        plt.title('Eccentricity' + ' - ' + str('RH'))
    if modality == 'meanbold':
        plt.ylim([0, .3])
        plt.title('Mean BOLD' + ' - ' + str('RH'))
    if modality == 'curvature':
        plt.ylim([0, .3])
        plt.title('Curvature' + ' - ' + str('RH'))
    sns.despine()

    # Create an output folder if it doesn't already exist
    directory = './../figures/figure2'
    if not osp.exists(directory):
        os.makedirs(directory)

    plt.savefig(
        './../figures/figure2/MeanDifFromTheMean_perHemi_' + str(modality) +
        '_181participants.pdf',
        format="pdf")
    return plt.show()


def intraIndividual_difference_plots_sameHemi(data, modality):
    """Generate dorsal versus ventral plots of intra individual variability
        of topographic features.

    Args:
        data (pandas DataFrame):  summary data used to generate the plots,
            which is the output of intraIndividual_difference_plots
        modality (string): modality of the data ('polarAngle', 'eccentricity')
    Returns:
        plt.show(): plot of the data

    """
    fig = plt.figure(figsize=(10, 5))
    fig.add_subplot(1, 2, 1)
    palette_2 = [sns.color_palette("PRGn_r")[5],
                 sns.color_palette("PRGn_r")[4]]
    sns.violinplot(x="Visual area",
                   y="Mean intra-individual variability in pRF estimates",
                   hue="Half", data=data[data['Hemisphere'] == 'LH'],
                   palette=palette_2, split=True, inner="quartile")
    if modality == 'polarAngle':
        plt.ylim([0, 65])
        plt.title('Polar angle' + ' - ' + str('LH'))
    if modality == 'eccentricity':
        plt.ylim([0, 3])
        plt.title('Eccentricity' + ' - ' + str('LH'))
    sns.despine()

    fig.add_subplot(1, 2, 2)
    palette_1 = [sns.color_palette("PRGn_r")[0],
                 sns.color_palette("PRGn_r")[1]]

    sns.violinplot(x="Visual area",
                   y="Mean intra-individual variability in pRF estimates",
                   hue="Half", data=data[data['Hemisphere'] == 'RH'],
                   palette=palette_1, split=True, inner="quartile")
    if modality == 'polarAngle':
        plt.ylim([0, 65])
        plt.title('Polar angle' + ' - ' + str('RH'))
    if modality == 'eccentricity':
        plt.ylim([0, 3])
        plt.title('Eccentricity' + ' - ' + str('RH'))
    sns.despine()

    # Create an output folder if it doesn't already exist
    directory = './../figures/supplementary_figure1'
    if not osp.exists(directory):
        os.makedirs(directory)

    plt.savefig(
        './../figures/supplementary_figure1/IntraIndividual_MeanDifFit2vsFit3_perHemi_' + str(
            modality) +
        '_181participants.pdf',
        format="pdf")
    return plt.show()
