from functions.individual_variability import difference_plots
from functions.individual_variability import difference_plots_sameHemi

path = './../data'

# All individuals
with open('./../list_subj') as fp:
    subjects = fp.read().split("\n")
list_of_ind = subjects[0:len(subjects) - 1]

# modalities = ['polarAngle', 'eccentricity', 'myelin', 'curvature']
modalities = ['polarAngle', 'eccentricity']

for i in range(len(modalities)):
    # data = difference_plots(modalities[i], 'list1')
    # data.to_pickle('data_list1_' + str(modalities[i]) + '.pkl')
    data = difference_plots(modalities[i], list_of_ind)
    data.to_pickle('./../output/longFormat_' + modalities[i] + '_all.pkl')
    data.to_excel('./../output/longFormat_' + modalities[i] + '_all.xlsx')
    difference_plots_sameHemi(data, modalities[i], 'list')
