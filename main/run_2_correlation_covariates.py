import numpy as np
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os.path as osp
import os

sys.path.append('..')

from functions.def_ROIs_WangParcels import roi
from functions.def_ROIs_EarlyVisualAreas import roi as ROI
from functions.individual_variability import grab_data
from scipy.stats import pearsonr

def correlation_cov(hemisphere):

    ### General settings ###
    # All individuals
    with open('./../list_subj.txt') as fp:
        subjects = fp.read().split("\n")
    list_of_ind = subjects[0:len(subjects) - 1]

    data = {}

    # Early visual areas
    final_mask_L_ROI, final_mask_R_ROI, index_L_mask_ROI, index_R_mask_ROI = \
        ROI(['ROI'])

    if hemisphere == 'LH':
        eccentricity_mask = np.reshape(
            np.load('./MaskEccentricity_'
                    'above1below8ecc_LH.npz')['list'], (-1))
    else:
        eccentricity_mask = np.reshape(
            np.load('./MaskEccentricity_'
                    'above1below8ecc_RH.npz')['list'], (-1))

    modalities = ['polarAngle', 'eccentricity', 'curvature', 'meanbold']

    ### Early visual areas - V1-3 ###
    for modality in modalities:
        data[modality] = []

        for subject in list_of_ind:
            if hemisphere == 'LH':
                tmp = grab_data(modality, subject, final_mask_L_ROI, 'LH')[
                    np.reshape(eccentricity_mask, (-1, 1)) != 0]
            else:
                tmp = grab_data(modality, subject, final_mask_R_ROI, 'RH')[
                    np.reshape(eccentricity_mask, (-1, 1)) != 0]

            data[modality].append(np.reshape(tmp, (-1, 1)))

        # Concatenate participants data
        data[modality] = np.concatenate(data[modality], axis=0)

    # Transform polar angle range
    data['polarAngle'] = np.abs(data['polarAngle'] - 180)

    # Reshape data
    for key in data:
        data[key] = np.reshape(data[key], (-1))

    # Stats
    for i in range(4):
        for j in range(4):
            if i != j:
                print(hemisphere + ': Correlation between ' + str(modalities[i]) + ' and ' + str(modalities[j]) +
                      ' is %s and p-value %s' % pearsonr(data[modalities[i]], data[modalities[j]]))

    # Correlation matrix plot
    df = pd.DataFrame.from_dict(data)

    corr_matrix = df.corr()

    # Create an output folder if it doesn't already exist
    directory = './../figures/figure3'
    if not osp.exists(directory):
        os.makedirs(directory)

    # Left hemisphere
    mask = np.zeros_like(corr_matrix)
    mask[np.triu_indices_from(mask)] = True

    sns.heatmap(corr_matrix, annot=corr_matrix, fmt='.2f',
                mask=mask, square=True, cmap="PuOr", vmin=-.5, vmax=.5)
    plt.savefig(
        './../figures/figure3/correlationNonNeuralVariables_' + hemisphere + '_V1-3_181participants.pdf', format="pdf")
    print('Correlation matrix "correlationNonNeuralVariables_' + hemisphere + '_V1-3_181participants.pdf" plot ' +
          'saved in ./../figures/figure3')
    print('_____________________________________________________________')
    plt.close()

    # To save as csv files
    directory = './../output/figure3'
    if not osp.exists(directory):
        os.makedirs(directory)

    corr_matrix.to_csv(
        './../output/figure3/correlationNonNeuralVariables_' + hemisphere + '_V1-3_181participants.csv')

    ### Per visual area ###
    areas = [['V1d'], ['V2d'], ['V3d'], ['V1v'], ['V2v'], ['V3v'], ['hV4']]
    for area in areas:
        data = {}

        # Mask for specific visual areas
        final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
            area)

        # Selecting only vertices of final_mask_L within the visual cortex (ROI)
        final_mask_L = final_mask_L[final_mask_L_ROI == 1]
        final_mask_R = final_mask_R[final_mask_R_ROI == 1]

        # Final mask - values equal to 2 mean nodes with visual area and within
        # eccentricity range
        if hemisphere == 'LH':
            mask = final_mask_L + eccentricity_mask
        else:
            mask = final_mask_R + eccentricity_mask

        area = area[0]
        modalities = ['polarAngle', 'eccentricity', 'curvature', 'meanbold',]
        for modality in modalities:
            data[modality] = []

            for subject in list_of_ind:
                if hemisphere == 'LH':
                    tmp = grab_data(modality, subject, final_mask_L_ROI, 'LH')[
                        np.reshape(mask, (-1, 1)) == 2]
                else:
                    tmp = grab_data(modality, subject, final_mask_R_ROI, 'RH')[
                        np.reshape(mask, (-1, 1)) == 2]
                data[modality].append(np.reshape(tmp, (-1, 1)))

            data[modality] = np.concatenate(data[modality], axis=0)

        # Transform polar angle range
        data['polarAngle'] = np.abs(data['polarAngle'] - 180)

        for key in data:
            data[key] = np.reshape(data[key], (-1))

        df = pd.DataFrame.from_dict(data)

        corr_matrix = df.corr()

        # Save as csv files
        corr_matrix.to_csv(
            './../output/figure3/correlationNonNeuralVariables_' + hemisphere + '_' + str(area) + '_181participants.csv')

        # Correlation matrix plot - Left hemisphere
        mask = np.zeros_like(corr_matrix)
        mask[np.triu_indices_from(mask)] = True

        sns.heatmap(corr_matrix, annot=corr_matrix, fmt='.2f', mask=mask, square=True,
                    cmap="PuOr", vmin=-.5, vmax=.5)
        plt.savefig('./../figures/figure3/correlationNonNeuralVariables_' + hemisphere + '_' + str(area) +
                    '_181participants.pdf', format="pdf")
        print('Correlation matrix "correlationNonNeuralVariables_' + hemisphere + '_' + str(area) +
              '_181participants.pdf" saved in ./../output/figure3')
        print('_____________________________________________________________')
        plt.close()
    return print('Correlation matrices of the ' + hemisphere + ' are complete')


if __name__ == '__main__':
    correlation_cov('LH')
    correlation_cov('RH')
