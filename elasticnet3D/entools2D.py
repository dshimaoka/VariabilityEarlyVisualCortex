#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 12:52:53 2023

created from elastic_sweep_slurm.py

@author: daisuke
"""

#############################
# v2 geometry
# specific to marmoset V2
#############################

import numpy as np
from scipy.special import erfinv
import matplotlib.pyplot as plt
import functions.dstools as dst
import astropy.stats as ast

# x is cortical space in mm
# it should go from 0.0mm to 8mm, which should give eccentricity to 0.8 to 10.0
# this is derived from the cortical magnification function of V2
# ie. first take the integral of the cortical magnification function, then make inverse function
# note: this is too much magnification. Use the version below
def x2ecc_old(x):
    return np.exp(1.7162 + 3.6761 * erfinv(-0.5449 + 0.092 * x))

# x is cortical space in mm
# it should go from 0.0mm to 5.5mm, which should give eccentricity to 2.0 to 10.0
# this is derived from the cortical magnification function of V2
# ie. first take the integral of the cortical magnification function, then make inverse function
def x2ecc(x):
    return np.exp(1.7162 + 3.6761 * erfinv(-0.306111 + 0.0923636 * x))

# generate the visual field coordinates of the rostral boundary of V2
# the length ogf the boundary is boundary_len mm,
# which is divided into div divisions
def generate_v2_boundary(boundary_len, div):
    zz = []
    for x in np.linspace(0.0, boundary_len, div):
        # horizontal meridian
        coords = [x2ecc(x), 0.0]
        zz.append(coords)
    return np.array(zz)

#############################
# gemoetry
#############################

### generate landmarks on the visual field
def spiral(c, r0, r1, n, shuffleP=True):
    """
        Make a spiral with n points.
        the distance from these points to (0, 0) ranges from r0 to r1
        the density of the dots falls off accoridng to r^c, where r is the distance to (0, 0)
        c should be a negative number
        returns a numpy array
    """
    def f(r):
        return np.sqrt(np.power(r, 1.0-c)/(1.0-c))

    n0 = np.power(r0 * r0 * (1.0-c), 1.0/(1.0-c))
    n1 = np.power(r1 * r1 * (1.0-c), 1.0/(1.0-c))

    lst = []
    for k in range(1, n+1):
        d = f( (n1-n0)/(n-1)*k + n0-(n1-n0)/(n-1) )
        theta = 3.1415926 * (3.0 - np.sqrt(5.0)) * (k-1.0)
        x = d * np.cos(theta)
        y = d * np.sin(theta)
        lst.append((x,y))
    res = np.asarray(lst, dtype=np.float64)
    if shuffleP:
        np.random.shuffle(res)
    return res

def generate_landmarks(c, r0, r1, n, noise=False):
    # generate landmarks on a hemifield
    # returns a numpy array

    # generate twice as many points because spiral() covers the entire visual field
    pts = spiral(c, r0, r1, 2*n)

    # boolean array, find those that are in the right hemifield
    pts_idx = pts[:, 0] >= 0.0

    # make a mask array
    pts_idx = np.stack([pts_idx, pts_idx])
    pts_idx = np.transpose(pts_idx)

    # keep points whose x coordinate is larger or equal to 0
    zz = np.extract(pts_idx, pts)
    # but since the extracted is flattend, I have to reshape it
    zz = zz.reshape([int(zz.shape[0]/2), 2])

    if noise:
        zz = zz + np.random.normal(0.0, 0.1, zz.shape)

    return zz

def generate_landmarks_fromData(retinotopy, n, noise=False):
    # generate landmarks on a hemifield
    # returns a numpy array

    # generate twice as many points because spiral() covers the entire visual field
    pts = retinotopy 

    # boolean array, find those that are in the right hemifield
    #pts_idx = pts[:, 0] >= 0.0
    pts_idx_ori = np.arange(0,len(pts))

    # Draw values randomly from the array 10 times with replacement
    pts_idx = np.random.choice(pts_idx_ori, size=n, replace=True)

       # make a mask array
    #pts_idx = np.stack([pts_idx, pts_idx])
    #pts_idx = np.transpose(pts_idx)


    # keep points whose x coordinate is larger or equal to 0
    #zz = np.extract(pts_idx, pts)
    zz = pts[pts_idx,:]
    
    # but since the extracted is flattend, I have to reshape it
    #zz = zz.reshape([int(zz.shape[0]/2), 2])

    if noise:
        zz = zz + np.random.normal(0.0, 0.1, zz.shape)

    return zz


def neighborhood(h, w):
    """
        numbers {0...h*w-1} arranged in a hxw grid, produce a list of neighboring numbers.
        The ith element of the list is a list of numbers that neighbor i
    """
    # this line defines how the geometry is coded
    # for (h, w) = (4, 2)
    # 0 4
    # 1 5
    # 2 6
    # 3 7
    x = np.transpose(np.array(range(h*w)).reshape([w, h]))
    lst = []
    for i in range(w):
        for j in range(h):
            neighbors = []
            for ii in [-1, 0, 1]:
                for jj in [-1, 0, 1]:
                    if (i+ii>=0) and (i+ii<w) and (j+jj>=0) and (j+jj<h):
                        neighbors.append(x[j+jj,i+ii])
            lst.append(neighbors)
    return lst

def make_mask(h, w, neighbors):
    x = np.zeros([h*w, h*w], dtype=np.float64)
    for i in range(h*w):
        for j in neighbors[i]:
            x[i, j] = 1.0
    return x

def make_boundary_mask(n):
    """
        n: numer of elements on the boundary
        return a n*n array
    """
    x = np.zeros([n, n], dtype=np.float64)
    for i in range(n):
        for j in [-1, 0, 1]:
            if (i+j)>=0 and (i+j)<n:
                x[i, i+j] = 1.0
    return x

def initial_condition(map_size, x):
    """
        initialze a cortical map of the size [map_w * map_h]
        each point on the map is randomly assigned to prototypes on the visual field (x) with some added noise
    """
    #map_size = h * w

    idx = np.random.randint(x.shape[0], size = map_size)
    y = x[idx, :] # randomly sample from x with replacement
    n = np.random.normal(0.0, 1.0, y.shape) # the standard deviation is set to 5.0 deg
    return y + n


# def getNode(graph, h, w)
#     """
# 	obtain node in DM
#     """

# def getNode_vf(graph, x, y)
#     """
# 	obtain node in V2 whose preferred position is closest to y
# 	r: retinotopy in V2 corresponding to each node in graph [x, y]
# 	y: retinotopy in DM [x, y]
#     """

def showCostFuncTrajectory(reg1_all, reg2_all, thisDir, suffix):
    plt.semilogy(reg1_all, label='reg1');
    plt.semilogy(reg2_all, label='reg2');
    plt.legend()
    plt.xlabel('iteration')
    plt.gcf().set_size_inches(10, 10)
    plt.savefig(thisDir+'/costFunction_'+suffix, dpi=200)

def resultSummary(result, yb, retinotopy, map_h, map_w, mask_idx, mask_var_idx, mask_var_sub, mask_fix_idx, mask_fix_sub, thisDir, suffix, subject_id):
  

        #convert from cartesian to polar coordinates
        result_ecc, result_pa = dst.cartesian_to_polar(result[:,0],result[:,1])
        result_pol = np.column_stack((result_ecc,result_pa)) #[ecc, pa]
        
        yb_ecc, yb_pa = dst.cartesian_to_polar(yb[:,0],yb[:,1])
        yb_pol = np.column_stack((yb_ecc,yb_pa)) #[ecc, pa]
        
        retinotopy_ecc, retinotopy_pa = dst.cartesian_to_polar(retinotopy[:,0],retinotopy[:,1])
        retinotopy_pol = np.column_stack((retinotopy_ecc,retinotopy_pa)) #[ecc, pa] in [deg]
        
        # result in cartesian coordinate
        result2d = np.nan * np.ones((map_h,map_w,2))
        for pp in range(0,len(mask_var_idx)):
            result2d[mask_var_sub[pp,1],mask_var_sub[pp,0],:] = result[pp,:]
        for qq in range(0,len(mask_fix_idx)):
            result2d[mask_fix_sub[qq,1],mask_fix_sub[qq,0],:] = yb[qq,:]
        plt.subplot(121); plt.imshow(result2d[:,:,0].T, origin='lower'); plt.colorbar(); plt.title('simulated azimuth')
        plt.subplot(122); plt.imshow(result2d[:,:,1].T, origin='lower'); plt.colorbar(); plt.title('simulated altitude')
        plt.draw()
        plt.gcf().set_size_inches(20, 10)
        plt.savefig(thisDir+'/simulated_retinotopy_cartesian_'+suffix, dpi=200)
        
        # result in polar coordinate
        result2d_pol = np.nan * np.ones((map_h,map_w,2))
        for pp in range(0,len(mask_var_idx)):
            result2d_pol[mask_var_sub[pp,1],mask_var_sub[pp,0],:] = result_pol[pp,:]
        for qq in range(0,len(mask_fix_idx)):
            result2d_pol[mask_fix_sub[qq,1],mask_fix_sub[qq,0],:] = yb_pol[qq,:]
        plt.subplot(121); plt.imshow(result2d_pol[:,:,0].T, origin='lower'); plt.title('simulated eccentricity')
        plt.subplot(122); plt.imshow(result2d_pol[:,:,1].T, origin='lower', vmin=0, vmax=361, cmap='gist_rainbow_r'); plt.title('simulated polar angle')
        plt.draw()
        plt.gcf().set_size_inches(20, 10)
        plt.savefig(thisDir+'/simulated_retinotopy_polar_'+suffix, dpi=200)
        
        # initial condition
        # init2d = np.nan * np.ones((map_h,map_w,2))
        # for pp in range(0,len(mask_var_idx)):
        #     init2d[mask_var_sub[pp,1],mask_var_sub[pp,0],:] = y0[pp,:]
        # for qq in range(0,len(mask_fix_idx)):
        #     init2d[mask_fix_sub[qq,1],mask_fix_sub[qq,0],:] = yb[qq,:]
        # plt.subplot(121); plt.imshow(init2d[:,:,0].T, origin='lower'); plt.colorbar(); plt.title('azimuth')
        # plt.subplot(122); plt.imshow(init2d[:,:,1].T, origin='lower'); plt.colorbar(); plt.title('altitude')

        # original data in cartesian coordinate
        orig2d = np.nan * np.ones((map_h,map_w,2))
        for pp in range(0,len(mask_var_idx)):
            orig2d[mask_var_sub[pp,1],mask_var_sub[pp,0],:] = retinotopy[mask_var_idx[pp],:]
        for qq in range(0,len(mask_fix_idx)):
            orig2d[mask_fix_sub[qq,1],mask_fix_sub[qq,0],:] = retinotopy[mask_fix_idx[qq],:]
        plt.subplot(121); plt.imshow(orig2d[:,:,0].T, origin='lower'); plt.colorbar(); plt.title('azimuth')
        plt.subplot(122); plt.imshow(orig2d[:,:,1].T, origin='lower'); plt.colorbar(); plt.title('altitude')
        plt.draw()
        plt.gcf().set_size_inches(20, 10)
        plt.savefig(thisDir+'/original_retinotopy_cartesian_' + subject_id, dpi=200)

        #original data in polar coordinate
        orig2d_pol = np.nan * np.ones((map_h,map_w,2))
        for pp in range(0,len(mask_var_idx)):
            orig2d_pol[mask_var_sub[pp,1],mask_var_sub[pp,0],:] = retinotopy_pol[mask_var_idx[pp],:]
        for qq in range(0,len(mask_fix_idx)):
            orig2d_pol[mask_fix_sub[qq,1],mask_fix_sub[qq,0],:] = retinotopy_pol[mask_fix_idx[qq],:]
        plt.subplot(121); plt.imshow(orig2d_pol[:,:,0].T, origin='lower'); plt.title('eccentricity')
        plt.subplot(122); plt.imshow(orig2d_pol[:,:,1].T, origin='lower', vmin=0, vmax=361, cmap='gist_rainbow_r'); plt.title('polar angle')
        plt.draw()
        plt.gcf().set_size_inches(20, 10)
        plt.savefig(thisDir+'/original_retinotopy_polar_' + subject_id, dpi=200)
        plt.close()

        # deviance between original and simulation in [deg]
        #dev2d = np.sqrt((result2d[:,:,0]-orig2d[:,:,0])**2 + (result2d[:,:,1]-orig2d[:,:,1])**2)
        #plt.imshow(dev2d.T, origin='lower'); plt.colorbar(); plt.title('deviance [deg]')

         # correlation between original and simulation
        corr_azimuth = np.corrcoef(retinotopy[mask_var_idx,0], result[:,0])[0,1];
        corr_altitude = np.corrcoef(retinotopy[mask_var_idx,1], result[:,1])[0,1];

        # circular correlation of PA
        corr_pa = ast.circcorrcoef(np.pi/180*retinotopy_pol[mask_var_idx, 1], np.pi/180*result_pa)

        #mask_var_idx = np.concatenate([mask_idx[index] for index in varIdx])
        #corr_azimuth[thisArea] = np.corrcoef(retinotopy[mask_idx[index] for index in thisArea,0], result[:,0])[0,1];
        corr_azimuth_v2 = np.corrcoef(retinotopy[mask_idx[1],0], result[0:len(mask_idx[1]),0])[0,1];
        corr_altitude_v2 = np.corrcoef(retinotopy[mask_idx[1],1], result[0:len(mask_idx[1]),1])[0,1];
        
        return corr_azimuth, corr_altitude, corr_pa
    
    