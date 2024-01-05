#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 13:19:10 2024

@author: daisuke
"""
import numpy as np
import functions.dstools as dst #FIXME
import matplotlib.pyplot as plt

sweep_id = 0
out_root_dir = "/tmp/";

#TMP
# map_h = 30
# map_w = 15
# mask_v2_d_idx = np.arange(0,map_h*map_w)
# a,b = dst.ind2sub([map_w, map_h], mask_v2_d_idx)
# mask_v2_d_sub = np.column_stack((a,b)).astype(int) #[x,y]
#


out_dir = out_root_dir + 'sweep_' + str(sweep_id).zfill(2) + "/"

#for i1 in range(0, 16):
#    for i2 in range(0, 16):
    
i1=0;
i2=4;
b1 = 0.001*1.6**i1
b2 = 0.001*1.6**i2

saveName = out_dir + "y-" + "%6.5f"%b1 + '-' + "%6.5f"%b2 + ".data"
data = np.loadtxt(saveName)

result2d = np.nan * np.ones((map_h,map_w,2))
for pp in range(0,len(mask_v2_d_idx)):
    result2d[mask_v2_d_sub[pp,1],mask_v2_d_sub[pp,0],:] = data[pp,:]
for qq in range(0,len(mask_v1_d_idx)):
    result2d[mask_v1_d_sub[qq,1],mask_v1_d_sub[qq,0],:] = yb[qq,:]
plt.subplot(121); plt.imshow(result2d[:,:,0].T, origin='lower'); plt.colorbar(); plt.title('azimuth')
plt.subplot(122); plt.imshow(result2d[:,:,1].T, origin='lower'); plt.colorbar(); plt.title('altitude')


# initial condition
init2d = np.nan * np.ones((map_h,map_w,2))
for pp in range(0,len(mask_v2_d_idx)):
    init2d[mask_v2_d_sub[pp,1],mask_v2_d_sub[pp,0],:] = y0[pp,:]
for qq in range(0,len(mask_v1_d_idx)):
    init2d[mask_v1_d_sub[qq,1],mask_v1_d_sub[qq,0],:] = yb[qq,:]
plt.subplot(121); plt.imshow(init2d[:,:,0].T, origin='lower'); plt.colorbar(); plt.title('azimuth')
plt.subplot(122); plt.imshow(init2d[:,:,1].T, origin='lower'); plt.colorbar(); plt.title('altitude')


# original retinotopy
orig2d = np.nan * np.ones((map_h,map_w,2))
for pp in range(0,len(mask_v2_d_idx)):
    orig2d[mask_v2_d_sub[pp,1],mask_v2_d_sub[pp,0],:] = retinotopy[mask_v2_d_idx[pp],:]
for qq in range(0,len(mask_v1_d_idx)):
    orig2d[mask_v1_d_sub[qq,1],mask_v1_d_sub[qq,0],:] = retinotopy[mask_v1_d_idx[qq],:]
plt.subplot(121); plt.imshow(orig2d[:,:,0].T, origin='lower'); plt.colorbar(); plt.title('azimuth')
plt.subplot(122); plt.imshow(orig2d[:,:,1].T, origin='lower'); plt.colorbar(); plt.title('altitude')
