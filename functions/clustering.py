import numpy as np
import sys

sys.path.append('..')

from functions.def_ROIs_EarlyVisualAreas import roi
from functions.individual_variability import grab_data

def averageMaps(path, list_of_ind, mask_L, mask_R, modality, cluster_index):
    '''
    Return average maps for a given ROI and modality
    Parameters
    ----------
    path : str
        Path to the data folder
    list_of_ind : list
        List of subjects
    mask_L : array
        Mask for the left hemisphere
    mask_R : array
        Mask for the right hemisphere
    modality : str
        Modality name
    cluster_index : str
        Cluster index
    Returns
    -------
    data : dict
        Dictionary containing the data for each hemispher
    '''

    data = {str(modality) + '_data_LH': [],
            str(modality) + '_data_RH': []}
    for i in range(len(list_of_ind)):
        data[str(modality) + '_data_LH'].append(
            grab_data(str(modality), str(list_of_ind[i]), mask_L,
                    'LH'))
        data[str(modality) + '_data_RH'].append(
            grab_data(str(modality), str(list_of_ind[i]), mask_R,
                    'RH'))

    hemispheres = ['LH', 'RH']
    for hemisphere in hemispheres:
        mean = np.mean(data[str(modality) + '_data_' + hemisphere], axis=0)
        np.savez('./../output/mean'+ str(modality) + '_PAclustering_cluster' + str(cluster_index) + '_' + hemisphere,
                list=mean)

if __name__ == '__main__':
    path = './../data'

    # Clustering assignment
    with open('./../list_subj.txt') as fp:
        subjects = fp.read().split("\n")
    list_of_subs = np.array(subjects[0:len(subjects) - 1])

    # Cluster assignment
    cluster_assignment = np.load('./../output/clusters_individualIndeces_PA.npz')['list']

    clusters = {}
    for i in np.unique(cluster_assignment):
        clusters['cluster_' + str(i + 1)] = list(list_of_subs[cluster_assignment == i])
    
    # ROI
    label_primary_visual_areas = ['ROI']
    final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
        label_primary_visual_areas)
    
    modalities = ['polarAngle','eccentricity', 'meanbold']

    for modality in modalities:
        for i in range(1, 7):
            averageMaps(path, clusters['cluster_' + str(i)], final_mask_L, final_mask_R, modality, i)