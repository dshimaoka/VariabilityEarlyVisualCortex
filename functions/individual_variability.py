import scipy
import os.path as osp
import os
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
            'myelin', 'curvature')
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

    if modality == 'curvature':
        data = \
            scipy.io.loadmat(
                osp.join(path, 'cifti_' + str(modality) + '_all.mat'))[
                'cifti_curv']

    else:
        data = \
            scipy.io.loadmat(
                osp.join(path, 'cifti_' + str(modality) + '_all.mat'))[
                'cifti_' + str(modality)]

    if hemisphere == 'LH':
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


def difference_score(modality, list_of_ind, list_of_areas):
    """Determine the vertex-wise difference between individual's data and the
    average map.

    Args:
        modality (string): modality of the data ('polarAngle', 'eccentricity',
            'myelin', 'curvature')
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


def difference_plots(modality, list_of_ind):
    """Generate left versus right hemispheres plots of individual variability
        of topographic features.

    Args:
        modality (string): modality of the data ('polarAngle', 'eccentricity',
            'myelin', 'curvature')
        list_of_ind (string): name of the list of HCP IDs ('list1' or 'list2')

    Returns:
        df (pandas DataFrame): summary data used to generate the plots
    """

    visual_areas = [['V1d'], ['V2d'], ['V3d']]

    fig = plt.figure(figsize=(10, 5))

    differences_dorsal = []
    for i in range(len(visual_areas)):
        diff_LH, diff_RH = difference_score(str(modality), list_of_ind,
                                            visual_areas[i])
        differences_dorsal_temp = np.concatenate(
            ([np.reshape(np.array(list_of_ind), (-1)),
              np.reshape(diff_LH, (-1)), len(diff_LH) * ['LH'],
              len(diff_LH) * visual_areas[i],
              len(diff_LH) * [visual_areas[i][0][:2]],
              len(diff_LH) * ['dorsal']],
             [np.reshape(np.array(list_of_ind), (-1)),
              np.reshape(diff_RH, (-1)), len(diff_RH) * ['RH'],
              len(diff_RH) * visual_areas[i],
              len(diff_RH) * [visual_areas[i][0][:2]],
              len(diff_RH) * ['dorsal']]), axis=1)
        differences_dorsal.append(differences_dorsal_temp)

    differences_dorsal_final = np.concatenate(differences_dorsal, axis=1)

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
    if modality == 'myelin':
        plt.ylim([0, .3])
        plt.title('Myelin')
    if modality == 'curvature':
        plt.ylim([0, .3])
        plt.title('Curvature')
    sns.despine()

    # Ventral areas
    visual_areas = [['V1v'], ['V2v'], ['V3v']]
    differences_ventral = []
    for i in range(len(visual_areas)):
        diff_LH, diff_RH = difference_score(str(modality), list_of_ind,
                                            visual_areas[i])
        differences_ventral_temp = np.concatenate(
            ([np.reshape(np.array(list_of_ind), (-1)),
              np.reshape(diff_LH, (-1)), len(diff_LH) * ['LH'],
              len(diff_LH) * visual_areas[i],
              len(diff_LH) * [visual_areas[i][0][:2]],
              len(diff_LH) * ['ventral']],
             [np.reshape(np.array(list_of_ind), (-1)),
              np.reshape(diff_RH, (-1)), len(diff_RH) * ['RH'],
              len(diff_RH) * visual_areas[i],
              len(diff_RH) * [visual_areas[i][0][:2]],
              len(diff_RH) * ['ventral']]), axis=1)
        differences_ventral.append(differences_ventral_temp)

    differences_ventral_final = np.concatenate(differences_ventral, axis=1)

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
    if modality == 'myelin':
        plt.ylim([0, .3])
        plt.title('Myelin')
    if modality == 'curvature':
        plt.ylim([0, .3])
        plt.title('Curvature')
    sns.despine()

    # Create an output folder if it doesn't already exist
    directory = './../output'
    if not osp.exists(directory):
        os.makedirs(directory)


    plt.savefig('./../output/MeanDifFromTheMean_combined_' + str(modality) +
                '_181participants', format="svg")
    plt.show()
    df = pd.concat([df_1, df_2])
    return df


def difference_plots_sameHemi(data, modality, list_of_ind):
    """Generate dorsal versus ventral plots of individual variability
        of topographic features.

    Args:
        data (pandas DataFrame):  summary data used to generate the plots,
            which is the output of difference_plots
        modality (string): modality of the data ('polarAngle', 'eccentricity',
            'myelin', 'curvature')
        list_of_ind (string): name of the list of HCP IDs ('list1' or 'list2')

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
    if modality == 'myelin':
        plt.ylim([0, .3])
        plt.title('Myelin' + ' - ' + str('LH'))
    if modality == 'curvature':
        plt.ylim([0, .3])
        plt.title('Curvature' + ' - ' + str('LH'))
    # plt.title(str(title) + ' - ' + str('LH'))
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
    if modality == 'myelin':
        plt.ylim([0, .3])
        plt.title('Myelin' + ' - ' + str('RH'))
    if modality == 'curvature':
        plt.ylim([0, .3])
        plt.title('Curvature' + ' - ' + str('RH'))
    # plt.title(str(title) + ' - ' + str('RH'))
    sns.despine()

    # Create an output folder if it doesn't already exist
    directory = './../output'
    if not osp.exists(directory):
        os.makedirs(directory)

    plt.savefig(
        './../output/MeanDifFromTheMean_perHemi_' + str(modality) +
        '_181participants',
        format="svg")
    plt.show()
