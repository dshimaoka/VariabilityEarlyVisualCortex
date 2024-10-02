#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 13:50:19 2024

@author: daisuke
"""
import os
os.chdir('/home/daisuke/Documents/git/VariabilityEarlyVisualCortex');

import os.path as osp
import pymeshlab
import functions.dstools as dst

loadDir = '/mnt/dshi0006_market/VariabilityEarlyVisualCortex/';
all_ids = ['134627', '155938', '193845', '200210', '318637'];
#all_ids = dst.getSubjectId('/home/daisuke/Documents/git/VariabilityEarlyVisualCortex/data/cifti_polarAngle_all.mat')



for ids in range(0,len(all_ids)):
    subject_id = all_ids[ids]
    thisDir = osp.join(loadDir, str(subject_id))
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(osp.join(thisDir,'Geom_' + str(subject_id) + '.stl'))
    ms.apply_coord_hc_laplacian_smoothing()
    ms.apply_coord_hc_laplacian_smoothing() #apply twice for 114823
    #ms.apply_coord_hc_laplacian_smoothing() #apply thrice for 134627, 193845, 200210, 
    #NG after applying 3 times! 155938, 318637
    ms.save_current_mesh(osp.join(thisDir, 'Geom_' + str(subject_id) + '_hclaplacian.stl'))