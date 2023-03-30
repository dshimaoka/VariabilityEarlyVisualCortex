#%%
import numpy as np
import sys

sys.path.append('..')

from functions.individual_variability import intra_individual_difference_score
from functions.individual_variability import intraIndividual_difference_plots
from functions.individual_variability import intraIndividual_difference_plots_sameHemi

# All individuals
with open('./../list_subj') as fp:
    subjects = fp.read().split("\n")
list_of_ind = subjects[0:len(subjects) - 1]

modalities = ['polarAngle', 'eccentricity']
for modality in modalities:

    # Dorsal areas
    area_dorsal = [['V1d'], ['V2d'], ['V3d']]
    differences_dorsal = []
    for area in area_dorsal:
        diff_LH, diff_RH = intra_individual_difference_score(str(modality), list_of_ind,
                                            area)
        differences_dorsal_temp = np.concatenate(
            ([np.reshape(np.array(list_of_ind), (-1)),
              np.reshape(diff_LH, (-1)), len(diff_LH) * ['LH'],
              len(diff_LH) * area,
              len(diff_LH) * [area[0][:2]],
              len(diff_LH) * ['dorsal']],
             [np.reshape(np.array(list_of_ind), (-1)),
              np.reshape(diff_RH, (-1)), len(diff_RH) * ['RH'],
              len(diff_RH) * area,
              len(diff_RH) * [area[0][:2]],
              len(diff_RH) * ['dorsal']]), axis=1)
        differences_dorsal.append(differences_dorsal_temp)

    differences_dorsal_final = np.concatenate(differences_dorsal, axis=1)

    # Ventral areas
    area_ventral = [['V1v'], ['V2v'], ['V3v']]
    differences_ventral = []
    for area in area_ventral:
        diff_LH, diff_RH = intra_individual_difference_score(str(modality), list_of_ind,
                                            area)
        differences_ventral_temp = np.concatenate(
            ([np.reshape(np.array(list_of_ind), (-1)),
              np.reshape(diff_LH, (-1)), len(diff_LH) * ['LH'],
              len(diff_LH) * area,
              len(diff_LH) * [area[0][:2]],
              len(diff_LH) * ['ventral']],
             [np.reshape(np.array(list_of_ind), (-1)),
              np.reshape(diff_RH, (-1)), len(diff_RH) * ['RH'],
              len(diff_RH) * area,
              len(diff_RH) * [area[0][:2]],
              len(diff_RH) * ['ventral']]), axis=1)
        differences_ventral.append(differences_ventral_temp)

    differences_ventral_final = np.concatenate(differences_ventral, axis=1)

    data = intraIndividual_difference_plots(modality, differences_dorsal_final, differences_ventral_final)
    data.to_pickle('./../output/lme/intraSubj_longFormat_' + modality + '_all.pkl')
    data.to_excel('./../output/lme/intraSubj_longFormat_' + modality + '_all.xlsx')
    intraIndividual_difference_plots_sameHemi(data, modality,)

# %%
