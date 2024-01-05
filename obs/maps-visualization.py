# -*- coding: utf-8 -*-
"""
2023-12-09 DS created from notebooks/maps-visualization.ipynb
"""

import numpy as np
#import pandas as pd
#import random
import sys
import os
import matplotlib.pyplot as plt
import scipy

sys.path.append('../')

from figures.polarAngle_maps_DS import polarAngle_plot
from functions.gradientAnalysis_DS import getVFSmap
#from ipywidgets import interact, Dropdown 


rootDir = '/home/daisuke/Documents/git/VariabilityEarlyVisualCortex';
saveDir = os.path.join(rootDir, 'results');

# All individuals
with open(os.path.join(rootDir,'list_subj.txt')) as fp:
    subjects = fp.read().split("\n")
list_of_subs = np.array(subjects[0:len(subjects) - 1])

# Cluster assignment
cluster_assignment = np.load(os.path.join(rootDir, 'output/clusters_individualIndeces_PA.npz'))['list']

clusters = {}
for i in np.unique(cluster_assignment):
    clusters['cluster_' + str(i + 1)] = list(list_of_subs[cluster_assignment == i])
    
cluster_index = None;#'cluster_4';#Dropdown(options = clusters.keys())
subject_index= ['146735','157336','585256','114823','581450','725751'];#Dropdown()
dorsal_roi = False;#Dropdown(options = [False, True])
binarize = False;#Dropdown(options = [False, True])
smoothing = True;

# retinotopy of individual subjects
for thisIndex in subject_index:
    #fig = polarAngle_plot(thisIndex, os.path.join(rootDir,'data'), dorsal_roi, 
    #                  binarize, save=True, save_path=saveDir);
    vfs,  grid_z0_PA, grid_z0_ecc = getVFSmap(thisIndex, os.path.join(rootDir,'data'), dorsal_only=dorsal_roi,
                      smoothing = smoothing, binarizing = binarize);
    if smoothing==False:
        vfsfilename = os.path.join(saveDir, 'fieldSign_' + str(thisIndex)  + '.mat')
    else:
        vfsfilename = os.path.join(saveDir, 'fieldSign_' + str(thisIndex)  + '_smoothed.mat')
    scipy.io.savemat(vfsfilename, {'vfs': vfs, 'grid_z0_PA': grid_z0_PA, 'grid_z0_ecc': grid_z0_ecc});

# curvature averaged within each cluster

 # # Loading the curvature map
 # curv = scipy.io.loadmat(osp.join(path, 'cifti_curvature_all.mat'))[
 #     'cifti_curvature']
 # background = np.reshape(
 #     curv['x' + subject_id + '_curvature'][0][0][0:32492], (-1))

 # # Background settings
 # threshold = 1  # threshold for the curvature map
 # nocurv = np.isnan(background)
 # background[nocurv == 1] = 0
 
 # nSubjects = 5;
 # background_all = np.zeros((32492, nSubjects));
 # background_all[0:32491] = background;
 
 
 # visual field sign

