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
