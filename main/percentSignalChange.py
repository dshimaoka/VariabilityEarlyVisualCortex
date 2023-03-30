#%%
import numpy as np
import sys

sys.path.append('..')

from functions.individual_variability import difference_plots
from functions.individual_variability import difference_plots_sameHemi
from functions.BOLDamplitude import BOLDamplitude

#%%
# All individuals
with open('./../list_subj') as fp:
    subjects = fp.read().split("\n")
list_of_ind = subjects[0:len(subjects) - 1]

#%%
# Dorsal areas
area_dorsal = [['V1d'], ['V2d'], ['V3d']]
dorsal = []
for area in area_dorsal:
    mean_LH , mean_RH  = BOLDamplitude(list_of_ind, area)
    dorsal_temp = np.concatenate(
        ([np.reshape(np.array(list_of_ind), (-1)),
            np.reshape(mean_LH, (-1)),
            len(mean_LH) * ['LH'],
            len(mean_LH) * area,
            len(mean_LH) * [area[0][:2]],
            len(mean_LH) * ['dorsal']],
            [np.reshape(np.array(list_of_ind), (-1)),
            np.reshape(mean_RH, (-1)), 
            len(mean_RH) * ['RH'],
            len(mean_RH) * area,
            len(mean_RH) * [area[0][:2]],
            len(mean_RH) * ['dorsal']]), axis=1)
    dorsal.append(dorsal_temp)

dorsal_final = np.concatenate(dorsal, axis=1)

# Ventral areas
area_ventral = [['V1v'], ['V2v'], ['V3v']]
ventral = []
for area in area_ventral:
    mean_LH , mean_RH = BOLDamplitude(list_of_ind, area)
    ventral_temp = np.concatenate(
        ([np.reshape(np.array(list_of_ind), (-1)),
            np.reshape(mean_LH, (-1)), 
            len(mean_LH) * ['LH'],
            len(mean_LH) * area,
            len(mean_LH) * [area[0][:2]],
            len(mean_LH) * ['ventral']],
            [np.reshape(np.array(list_of_ind), (-1)),
            np.reshape(mean_RH, (-1)), 
            len(mean_RH) * ['RH'],
            len(mean_RH) * area,
            len(mean_RH) * [area[0][:2]],
            len(mean_RH) * ['ventral']]), axis=1)
    ventral.append(ventral_temp)

ventral_final = np.concatenate(ventral, axis=1)
data = difference_plots('PercentSignalChange', dorsal_final, ventral_final)
data.to_pickle('./../output/longFormat_PercentSignalChange_MSMall_all.pkl')
data.to_excel('./../output/longFormat_PercentSignalChange_MSMall_all.xlsx')
difference_plots_sameHemi(data, 'PercentSignalChange',)
