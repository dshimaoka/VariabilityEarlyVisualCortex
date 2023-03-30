import numpy as np
import os.path as osp
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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
list_s = subjects[0:len(subjects) - 1]

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
for i in list_s:
    data = grab_data('polarAngle', i, final_mask_L_ROI, 'LH')
    dataset_orig.append(data[final_mask_L[final_mask_L_ROI == 1] == 1])
    # dataset_orig.append(data[mask_LH == 1])

    # Binarizing shifted values
    data[(data >= 180) & (data <= 225)] = 0
    data[(data > 225) & (data <= 360)] = 90
    data[(data >= 135) & (data < 180)] = 360
    data[(data >= 0) & (data < 135)] = 270

    data = data[mask_LH == 1]
    dataset_segmented.append(data)

#### Similarity metrics ####
# Jaccard similarity
jaccard_matrix = np.zeros((181, 181))
for i in range(181):
    for j in range(181):
        jaccard_1 = jaccard_score(dataset_segmented[i],
                                  dataset_segmented[j],
                                  average='weighted')
        jaccard_2 = jaccard_score(dataset_segmented[j],
                                  dataset_segmented[i],
                                  average='weighted')
        jaccard_matrix[i, j] = (jaccard_1 + jaccard_2) / 2

# Plot jaccard matrix
sns.heatmap(jaccard_matrix, cmap="flare_r")
plt.show()

#### Clustering ####
# Spectral clustering
n_cluster = 6
clustering = SpectralClustering(n_clusters=n_cluster,
                                random_state=123,
                                affinity='precomputed').fit_predict(
    jaccard_matrix)

####
# Reordering
ordering = np.array([])
for i in range(n_cluster):
    ordering = np.concatenate((ordering, np.where(clustering == i)[0]), axis=0)
ordering = ordering.astype(int)

cluster_matrix = np.ones((181, 181))
for i in range(n_cluster):
    indeces = np.where(clustering == i)[0]
    for j in indeces:
        for k in indeces:
            cluster_matrix[j][k] = cluster_matrix[j][k] * (i + 2)

#### Distributions
reordered_jaccard = jaccard_matrix[ordering].T[ordering].T
distributions = {'cluster': [], 'similarity': []}
count = 0
second_counter = 0
for i in range(n_cluster):
    cluster_size = sum(clustering == i)
    tmp_cluster = reordered_jaccard[count:count + cluster_size,
                  count:count + cluster_size]
    list_similarities = tmp_cluster[np.triu(tmp_cluster, k=1) > 0]
    distributions['cluster'].extend(
        ['Cluster_' + str(i + 1)] * len(list_similarities))
    distributions['similarity'].extend(list_similarities)

    second_counter += cluster_size
    print(count, 1)

    j = i + 1
    tmp_counter = second_counter
    while j < 6:
        cluster_size_other = sum(clustering == j)
        between_clusters_tmp = reordered_jaccard[
                               count:count + cluster_size,
                               tmp_counter:tmp_counter +
                                           cluster_size_other]
        distributions['cluster'].extend(
            ['Cluster_' + str(i + 1) + '_' + str(j + 1)] * len(
                between_clusters_tmp.flatten()))
        # distributions['cluster'].extend(
        #     ['Cluster_' + str(j + 1) + '_' + str(i + 1)] * len(
        #         between_clusters_tmp.flatten()))
        distributions['similarity'].extend(between_clusters_tmp.flatten())
        # distributions['similarity'].extend(between_clusters_tmp.flatten())
        tmp_counter += cluster_size_other
        print(tmp_counter)
        j += 1

    count += cluster_size

data = pd.DataFrame(distributions)



### Figure ###
# we generate a color palette with Seaborn.color_palette()


def color_pal(cluster):
    palette_tmp = sns.color_palette("PRGn_r")
    return palette_tmp[cluster]
reps = [5] + [4]*5 + [5] + [4]*4 + [5] + [4]*3 + [5] + [4]*2 + [5] + [4]*1
palette_purple = map(color_pal, reps)
palette_purple = list(palette_purple)

# pal = sns.color_palette(palette='flare_r', n_colors=22)

# in the sns.FacetGrid class, the 'hue' argument is the one that is the one
# that will be represented by colors with 'palette'
g = sns.FacetGrid(data, row='cluster', hue='cluster', aspect=15, height=0.75,
                  palette=palette_purple)

# then we add the densities kdeplots for each month
g.map(sns.kdeplot, 'similarity',
      bw_adjust=1, clip_on=False,
      fill=True, alpha=1, linewidth=1.5)

# here we add a white line that represents the contour of each kdeplot
g.map(sns.kdeplot, 'similarity',
      bw_adjust=1, clip_on=True,
      color="w", lw=2)

# here we add a horizontal line for each plot
g.map(plt.axhline, y=0,
      lw=2, clip_on=False)

# we loop over the FacetGrid figure axes (g.axes.flat) and add the month as
# text with the right color
# notice how ax.lines[-1].get_color() enables you to access the last line's
# color in each matplotlib.Axes
names = data['cluster'].unique()
means = [np.mean(data[data['cluster'] == name]['similarity']) for name in
         names]

for i, ax in enumerate(g.axes.flat):
    ax.text(0.1, 0.2, names[i],
            fontweight='bold', fontsize=15,
            color=ax.lines[1].get_color())
    ax.axvline(means[i], color='grey', ls='--', lw=1.5)

# we use matplotlib.Figure.subplots_adjust() function to get the subplots to
# overlap
g.fig.subplots_adjust(hspace=-0.3)

# eventually we remove axes titles, yticks and spines
g.set_titles("")
g.set(yticks=[])
g.despine(bottom=True, left=True)

plt.setp(ax.get_xticklabels(), fontsize=15, fontweight='bold')
plt.xlabel('Jaccard index', fontweight='bold', fontsize=15)
g.fig.suptitle('Jaccard score distributions within and between clusters',
               fontsize=20,
               fontweight=20)

g.savefig('figure7.pdf', height=10, width=5, dpi=500, bbox_inches='tight',
          pad_inches=0.5)
plt.show()

## Figure
fig, axs = plt.subplots(6, 6)
counter = 0
counter_snd = 0
for i in range(6):

    sns.displot(data=data[data['cluster'] == 'cluster_' + str(i)],
                x='similarity', ax=axs[i, i])
    if i != j:
        for j in range(6):
            sns.displot(data=data[
                data['cluster'] == 'cluster_' + str(i) + '_' + str(j)],
                        x='similarity', ax=axs[j, i])
