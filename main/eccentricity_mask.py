import numpy as np
import sys

sys.path.append('..')

from functions.def_ROIs_EarlyVisualAreas import roi
from functions.individual_variability import grab_data

path = './../data'

# All individuals
with open('./../list_subj') as fp:
    subjects = fp.read().split("\n")
list_of_ind = subjects[0:len(subjects) - 1]

# ROI settings
label_primary_visual_areas = ['ROI']
final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
    label_primary_visual_areas)

# Data
modality = 'eccentricity'

data = {str(modality) + '_data_LH': [],
        str(modality) + '_data_RH': []}
for i in range(len(list_of_ind)):
    data[str(modality) + '_data_LH'].append(
        grab_data(str(modality), str(list_of_ind[i]), final_mask_L,
                  'LH'))
    data[str(modality) + '_data_RH'].append(
        grab_data(str(modality), str(list_of_ind[i]), final_mask_R,
                  'RH'))

hemispheres = ['LH', 'RH']
for hemisphere in hemispheres:

    mean_ecc = np.mean(data['eccentricity_data_' + hemisphere], axis=0)

    # Mask for prediction errors
    ecc_1to8 = []
    for i in range(len(mean_ecc)):
        if mean_ecc[i][0] < 1 or mean_ecc[i][0] > 8:
            ecc_1to8.append(0)
        else:
            ecc_1to8.append(mean_ecc[i][0])
    ecc_1to8 = np.reshape(np.array(ecc_1to8), (-1))
    np.savez('./main/MaskEccentricity_above1below8ecc_' + hemisphere,
             list=ecc_1to8 > 0)
