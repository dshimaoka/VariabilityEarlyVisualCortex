#%%
import numpy as np
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append('..')

from functions.individual_variability import grab_data
from functions.def_ROIs_EarlyVisualAreas import roi as ROI
from functions.def_ROIs_WangParcels import roi

# All individuals
with open('./../list_subj') as fp:
    subjects = fp.read().split("\n")
list_of_ind = subjects[0:len(subjects) - 1]

#%%
def meanBOLD(list_of_ind, list_of_areas):

    # Mask for the early visual cortex - ROI
    final_mask_L_ROI, final_mask_R_ROI, index_L_mask_ROI, index_R_mask_ROI = \
        ROI(['ROI'])

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

    # Data
    data = {str(modality) + '_data_LH': [],
            str(modality) + '_data_RH': []}
    for i in range(len(list_of_ind)):
        data[str(modality) + '_data_LH'].append(
            grab_data(str(modality), str(list_of_ind[i]), final_mask_L_ROI,
                        'LH')[mask_LH==2])
        data[str(modality) + '_data_RH'].append(
            grab_data(str(modality), str(list_of_ind[i]), final_mask_R_ROI,
                        'RH')[mask_RH==2])

    mean_LH = np.mean(data[str(modality) + '_data_LH'], axis=1)
    mean_RH = np.mean(data[str(modality) + '_data_RH'], axis=1)
    return mean_LH, mean_RH

#%%
modality = 'meanbold'
# Dorsal areas
area_dorsal = [['V1d'], ['V2d'], ['V3d']]
meanBOLD_dorsal = []
for area in area_dorsal:
    meanBOLD_LH, meanBOLD_RH = meanBOLD(list_of_ind, area)
    print(meanBOLD_LH)
    print(np.shape(meanBOLD_LH))
    meanBOLD_dorsal_temp = np.concatenate(
        ([np.reshape(np.array(list_of_ind), (-1)),
            np.reshape(meanBOLD_LH, (-1)), len(meanBOLD_LH) * ['LH'],
            len(meanBOLD_LH) * area,
            len(meanBOLD_LH) * [area[0][:2]],
            len(meanBOLD_LH) * ['dorsal']],
            [np.reshape(np.array(list_of_ind), (-1)),
            np.reshape(meanBOLD_RH, (-1)), len(meanBOLD_RH) * ['RH'],
            len(meanBOLD_RH) * area,
            len(meanBOLD_RH) * [area[0][:2]],
            len(meanBOLD_RH) * ['dorsal']]), axis=1)
    meanBOLD_dorsal.append(meanBOLD_dorsal_temp)

meanBOLD_dorsal_final = np.concatenate(meanBOLD_dorsal, axis=1)

# Ventral areas
area_ventral = [['V1v'], ['V2v'], ['V3v']]
meanBOLD_ventral = []
for area in area_ventral:
    meanBOLD_LH, meanBOLD_RH = difference_score(str(modality), list_of_ind,
                                        area)
    meanBOLD_ventral_temp = np.concatenate(
        ([np.reshape(np.array(list_of_ind), (-1)),
            np.reshape(meanBOLD_LH, (-1)), len(meanBOLD_LH) * ['LH'],
            len(meanBOLD_LH) * area,
            len(meanBOLD_LH) * [area[0][:2]],
            len(meanBOLD_LH) * ['ventral']],
            [np.reshape(np.array(list_of_ind), (-1)),
            np.reshape(meanBOLD_RH, (-1)), len(meanBOLD_RH) * ['RH'],
            len(meanBOLD_RH) * area,
            len(meanBOLD_RH) * [area[0][:2]],
            len(meanBOLD_RH) * ['ventral']]), axis=1)
    meanBOLD_ventral.append(meanBOLD_ventral_temp)

meanBOLD_ventral_final = np.concatenate(meanBOLD_ventral, axis=1)

#DataFrame
df_1 = pd.DataFrame(
    columns=['HCP_ID', 'Mean BOLD signal',
                'Hemisphere',
                'Hemi visual area', 'Visual area', 'Half'],
    data=meanBOLD_dorsal_final.T)
df_1['Mean BOLD signal'] = df_1[
    'Mean BOLD signal'].astype(float)

df_2 = pd.DataFrame(
    columns=['HCP_ID', 'Mean BOLD signal',
                'Hemisphere',
                'Hemi visual area', 'Visual area', 'Half'],
    data=meanBOLD_ventral_final.T)
df_2['Mean BOLD signal'] = df_1[
    'Mean BOLD signal'].astype(float)

data = pd.concat([df_1, df_2])

# Create an output folder if it doesn't already exist
directory = './../output/lme'
if not osp.exists(directory):
    os.makedirs(directory)


data.to_pickle('./../output/lme/longFormat_meanBOLDsignal_MSMall_all.pkl')
data.to_excel('./../output/lme/longFormat_meanBOLDsignal_MSMall_all.xlsx')
# %%
