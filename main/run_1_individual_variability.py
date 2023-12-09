import numpy as np
import sys
import os
import os.path as osp

# sys.path.append('../')
sys.path.append('/home/daisuke/Documents/git/VariabilityEarlyVisualCortex/')

from functions.individual_variability import difference_plots_sameHemi
from functions.individual_variability import difference_plots
from functions.individual_variability import difference_score

# All individuals
with open('./../list_subj.txt') as fp:
    subjects = fp.read().split("\n")
list_of_ind = subjects[0:len(subjects) - 1]

modalities = ['polarAngle', 'eccentricity', 'curvature', 'meanbold',]
for modality in modalities:

    # Dorsal areas
    area_dorsal = [['V1d'], ['V2d'], ['V3d']]
    differences_dorsal = []
    for area in area_dorsal:
        diff_LH, diff_RH = difference_score(str(modality), list_of_ind,
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
        diff_LH, diff_RH = difference_score(str(modality), list_of_ind,
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
    data = difference_plots(
        modality, differences_dorsal_final, differences_ventral_final)

    # Create an output folder if it doesn't already exist
    directory = './../output/lme'
    if not osp.exists(directory):
        os.makedirs(directory)

    data.to_csv('./../output/lme/longFormat_' +
                   modality + '_MSMall_all.csv')
    data.to_excel('./../output/lme/longFormat_' +
                  modality + '_MSMall_all.xlsx')
    print('Saved long format data for ' + modality +
          ' in ./../output/lme/longFormat_' + modality + '_MSMall_all.xlsx')
    difference_plots_sameHemi(data, modality,)
