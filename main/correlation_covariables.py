#%%
import numpy as np
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os.path as osp

sys.path.append('..')

from functions.individual_variability import grab_data
from functions.def_ROIs_EarlyVisualAreas import roi as ROI
from functions.def_ROIs_WangParcels import roi

#%%
# All individuals
with open('./../list_subj') as fp:
    subjects = fp.read().split("\n")
list_of_ind = subjects[0:len(subjects) - 1]

data_LH = {}
data_RH = {}

# Early visual areas
final_mask_L_ROI, final_mask_R_ROI, index_L_mask_ROI, index_R_mask_ROI = \
        ROI(['ROI'])

# Mask for eccentricity range
eccentricity_mask_LH = np.reshape(
    np.load('./MaskEccentricity_'
            'above1below8ecc_LH.npz')['list'], (-1))
eccentricity_mask_RH = np.reshape(
    np.load('./MaskEccentricity_'
            'above1below8ecc_RH.npz')['list'], (-1))

#%%
modalities = ['polarAngle', 'eccentricity', 'curvature', 'meanbold', ]
for modality in modalities:
    data_LH[modality] = []
    data_RH[modality] = []
    
    for subject in list_of_ind:
        tmp_LH = grab_data(modality, subject, final_mask_L_ROI, 'LH')[np.reshape(eccentricity_mask_LH, (-1,1))!=0]
        tmp_RH = grab_data(modality, subject, final_mask_R_ROI, 'RH')[np.reshape(eccentricity_mask_RH, (-1,1))!=0]
        
        data_LH[modality].append(np.reshape(tmp_LH, (-1,1)))
        data_RH[modality].append(np.reshape(tmp_RH, (-1,1)))

        print(subject)
    # Concatenate participants data
    data_LH[modality] = np.concatenate(data_LH[modality], axis = 0)
    data_RH[modality] = np.concatenate(data_RH[modality], axis = 0)

# Transform polar angle range
data_LH['polarAngle'] = np.abs(data_LH['polarAngle'] - 180)
data_RH['polarAngle'] = np.abs(data_RH['polarAngle'] - 180)

# Correlation
for key in data_LH:
    data_LH[key] = np.reshape(data_LH[key], (-1))
    data_RH[key] = np.reshape(data_RH[key], (-1))
    print(key)


#%%
# Stats
for i in range(4):
    for j in range(4):
        if i!=j:
            print('LH: Correlation between ' + str(modalities[i]) + ' and ' + str(modalities[j]) +
            ' is %s and p-value %s' % pearsonr(data_LH[modalities[i]],data_LH[modalities[j]]))
#%%
for i in range(4):
    for j in range(4):
        if i!=j:
            print('RH: Correlation between ' + str(modalities[i]) + ' and ' + str(modalities[j]) +
            ' is %s and p-value %s' % pearsonr(data_RH[modalities[i]],data_RH[modalities[j]]))

#%%
## Correlation matrix plot
df_LH = pd.DataFrame.from_dict(data_LH)
df_RH = pd.DataFrame.from_dict(data_RH)

corr_matrix_LH = df_LH.corr()
corr_matrix_RH = df_RH.corr()

# Create an output folder if it doesn't already exist
directory = './../output/figure3'
if not osp.exists(directory):
    os.makedirs(directory)

# Left hemisphere
mask = np.zeros_like(corr_matrix_LH)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr_matrix_LH, annot=corr_matrix_LH, fmt ='.2f', mask=mask, square=True, cmap="PuOr", vmin=-.5, vmax=.5)
plt.savefig('./../output/figure3/correlationNonNeuralVariables_LH_V1-3_181participants', format="svg")
plt.close()

# Right hemisphere
mask = np.zeros_like(corr_matrix_RH)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr_matrix_RH, annot=corr_matrix_RH, fmt= '.2f', mask=mask, square=True, cmap="PuOr", vmin=-.5, vmax=.5)
plt.savefig('./../output/figure3/correlationNonNeuralVariables_RH_V1-3_181participants', format="svg")
plt.close()

# %%
# Per visual area
areas = [['V1d'], ['V2d'], ['V3d'], ['V1v'], ['V2v'], ['V3v']]
for area in areas:
    data_LH = {}
    data_RH = {}

    # Mask for specific visual areas
    final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
        area)

    # Selecting only vertices of final_mask_L within the visual cortex (ROI)
    final_mask_L = final_mask_L[final_mask_L_ROI == 1]
    final_mask_R = final_mask_R[final_mask_R_ROI == 1]

    # Final mask - values equal to 2 mean nodes with visual area and within
    # eccentricity range
    mask_LH = final_mask_L + eccentricity_mask_LH
    mask_RH = final_mask_R + eccentricity_mask_RH

    area=area[0]
    modalities = ['polarAngle', 'eccentricity', 'curvature', 'meanbold',]
    for modality in modalities:
        data_LH[modality] = []
        data_RH[modality] = []
        
        for subject in list_of_ind:
            tmp_LH = grab_data(modality, subject, final_mask_L_ROI, 'LH')[np.reshape(mask_LH, (-1,1))==2]
            tmp_RH = grab_data(modality, subject, final_mask_R_ROI, 'RH')[np.reshape(mask_RH, (-1,1))==2]
            
            data_LH[modality].append(np.reshape(tmp_LH, (-1,1)))
            data_RH[modality].append(np.reshape(tmp_RH, (-1,1)))

            print(subject)
        data_LH[modality] = np.concatenate(data_LH[modality], axis = 0)
        data_RH[modality] = np.concatenate(data_RH[modality], axis = 0)

    # Transform polar angle range
    data_LH['polarAngle'] = np.abs(data_LH['polarAngle'] - 180)
    data_RH['polarAngle'] = np.abs(data_RH['polarAngle'] - 180)

    for key in data_LH:
        data_LH[key] = np.reshape(data_LH[key], (-1))
        data_RH[key] = np.reshape(data_RH[key], (-1))
        print(key)
    df_LH = pd.DataFrame.from_dict(data_LH)
    df_RH = pd.DataFrame.from_dict(data_RH)

    corr_matrix_LH = df_LH.corr()
    corr_matrix_RH = df_RH.corr()

    # Left hemisphere
    mask = np.zeros_like(corr_matrix_LH)
    mask[np.triu_indices_from(mask)] = True

    sns.heatmap(corr_matrix_LH, mask=mask, square=True, cmap="PuOr", vmin=-.5, vmax=.5)
    plt.savefig('./../output/figure3/correlationNonNeuralVariables_LH_' + str(area) +
                '_181participants', format="svg")
    plt.close()

    # Right hemisphere
    mask = np.zeros_like(corr_matrix_RH)
    mask[np.triu_indices_from(mask)] = True

    sns.heatmap(corr_matrix_RH, mask=mask, square=True, cmap="PuOr", vmin=-.5, vmax=.5)
    plt.savefig('./../output/figure3/correlationNonNeuralVariables_RH_' + str(area) +
                '_181participants', format="svg")
    plt.close()
# %%
