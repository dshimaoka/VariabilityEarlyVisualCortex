#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 09:59:07 2023

@author: daisuke
"""


import neuropythy as ny
import matplotlib as mpl
import matplotlib.pyplot as plt

sub = ny.hcp_subject(157336)
lh = sub.lh
flatmap = lh.mask_flatmap('V1_label', map_right='right')

# If you want to extract the information about the flatmap, for example, the coordinates of the vertices, you can use the following code:
coordinates = flatmap.coordinates
connected_edges = flatmap.tess.indexed_edges
triangles = flatmap.tess.indexed_faces

# To know more about the different properties available per subject:
print(sub.lh.properties.keys())

# To generate a plot of the flatmap as mesh, you can use the following code:
tri = mpl.tri.Triangulation(
    flatmap.coordinates[0], flatmap.coordinates[1],
    flatmap.tess.indexed_faces.T)

(fig, ax) = plt.subplots(1,1, dpi=1024, figsize=(4,4))

ny.cortex_plot(flatmap, color='r', mask='V1_label', alpha=0.5)

# # You can instead plot the polar angle map, if you uncomment the following line:
# ny.cortex_plot(flatmap, color='prf_polar_angle', mask=('prf_variance_explained', 0.05, 1),
#                axes=ax, alpha=.3)

ax.triplot(tri, lw=.1)
ax.axis('equal')
plt.show()
