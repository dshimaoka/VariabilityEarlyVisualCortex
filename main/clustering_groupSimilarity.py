import numpy as np
import os.path as osp
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.path.append('..')

from numpy.random import seed
from sklearn.metrics import jaccard_score
from functions.def_ROIs_EarlyVisualAreas import roi as ROI
from functions.def_ROIs_DorsalEarlyVisualCortex import roi
from sklearn.cluster import SpectralClustering
from functions.individual_variability import grab_data
from scipy.spatial import distance

# All individuals
with open('./../list_subj') as fp:
    subjects = fp.read().split("\n")
list = subjects[0:len(subjects) - 1]

# Mask - Early visual cortex
final_mask_L_ROI, final_mask_R_ROI, index_L_mask_ROI, index_R_mask_ROI = \
    ROI(['ROI'])

# Mask - Dorsal early visual cortex
final_mask_L, final_mask_R, index_L_mask, index_R_mask = \
    roi(['ROI'])

# Mask for eccentricity range
eccentricity_mask_LH = np.reshape(
    np.load('./MaskEccentricity_'
            'above1below8ecc_LH.npz')['list'], (-1))

# Final mask
mask_LH = final_mask_L_ROI + final_mask_L
mask_LH[mask_LH != 2] = 0
mask_LH[mask_LH == 2] = 1
mask_LH = mask_LH[final_mask_L_ROI == 1] * eccentricity_mask_LH

# Import polar angle maps and concatenate maps
dataset_segmented = []
dataset_orig = []
for i in list:
    data = grab_data('polarAngle', i, final_mask_L_ROI, 'LH')
    dataset_orig.append(data[final_mask_L[final_mask_L_ROI==1] == 1])
    # dataset_orig.append(data[mask_LH == 1])

    # Binarizing shifted values
    data[(data >= 180) & (data <= 225)] = 0
    data[(data > 225) & (data <= 360)] = 90
    data[(data >= 135) & (data < 180)] = 360
    data[(data >= 0) & (data < 135)] = 270

    data = data[mask_LH == 1]
    dataset_segmented.append(data)

# Distance to average
clustering = np.load('./../clusters_individualIndeces_PA.npz')['list']
for i in range(6):
    mean_cluster = np.load('./../output/cluster_' + str(i) + '_weightedJaccard_eccentricityMask.npz')['list']

    # Binarizing shifted values
    mean_cluster[(mean_cluster >= 180) & (mean_cluster <= 225)] = 0
    mean_cluster[(mean_cluster > 225) & (mean_cluster <= 360)] = 90
    mean_cluster[(mean_cluster >= 135) & (mean_cluster < 180)] = 360
    mean_cluster[(mean_cluster >= 0) & (mean_cluster < 135)] = 270

    tmp = final_mask_L_ROI + final_mask_L
    tmp[tmp != 2] = 0
    tmp[tmp == 2] = 1
    mask = tmp[final_mask_L_ROI == 1] * eccentricity_mask_LH
    mask = mask[final_mask_L[final_mask_L_ROI==1]==1]

    mean_cluster = mean_cluster[mask == 1]

    jaccard_matrix = np.zeros((np.sum(clustering==i)))

    indeces = np.where(clustering==i)[0]

    for j in range(len(indeces)):
        jaccard_matrix[j] = jaccard_score(mean_cluster,
                                  dataset_segmented[j],
                                  average='weighted')
    # Maximum
    ind_index = list[indeces[np.where(jaccard_matrix==np.max(jaccard_matrix))][0]]
    print('Individual ID most similar to cluster ' + str(i)+ ' is %s.' % ind_index)


    # Minimum
    ind_index = list[indeces[np.where(jaccard_matrix==np.min(jaccard_matrix))][0]]
    print('Individual ID least similar to cluster ' + str(i)+ ' is %s.' % ind_index)
