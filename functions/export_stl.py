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

loadDir = '/mnt/dshi0006_market/VariabilityEarlyVisualCortex/';

all_ids = ['114823','157336','585256','114823','581450','725751'];


for ids in range(0,len(all_ids)):
    subject_id = all_ids[ids]
    thisDir = osp.join(loadDir, str(subject_id))
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(osp.join(thisDir,'Geom_' + str(subject_id) + '.stl'))
    ms.apply_coord_hc_laplacian_smoothing()
    ms.save_current_mesh(osp.join(thisDir, 'Geom_' + str(subject_id) + '_hclaplacian.stl'))