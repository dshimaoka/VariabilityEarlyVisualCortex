from functions.individual_variability import grab_data
from functions.def_ROIs_EarlyVisualAreas import roi
import numpy as np
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import os
import os.path as osp

sys.path.append('..')


def color_pal(cluster):
    palette_tmp = sns.color_palette("PRGn_r")
    return palette_tmp[cluster]


def ridgeline_plot(data):
    '''
    Plot the ridgeline plot
    Parameters
    ----------
    data : dict
        Dictionary containing within and between clusters jaccard scores
    Returns
    -------
    None.
    '''
    # This code is largely inspired from the following link: https://www.python-graph-gallery.com/ridgeline-graph-seaborn
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    
    # Color palette configuration
    reps = [5] + [4]*5 + [5] + [4]*4 + [5] + [4]*3 + [5] + [4]*2 + [5] + [4]*1
    palette_purple = map(color_pal, reps)
    palette_purple = list(palette_purple)

    # Plot
    g = sns.FacetGrid(data, row='cluster', hue='cluster', aspect=15, height=0.75,
                      palette=palette_purple)

    g.map(sns.kdeplot, 'similarity',
          bw_adjust=1, clip_on=False,
          fill=True, alpha=1, linewidth=1.5)

    g.map(sns.kdeplot, 'similarity',
          bw_adjust=1, clip_on=True,
          color="w", lw=2)

    g.map(plt.axhline, y=0,
          lw=2, clip_on=False)
    
    names = data['cluster'].unique()
    means = [np.mean(data[data['cluster'] == name]['similarity']) for name in
             names]

    for i, ax in enumerate(g.axes.flat):
        ax.text(0.1, 0.2, names[i],
                fontweight='bold', fontsize=15,
                color=ax.lines[1].get_color())
        ax.axvline(means[i], color='black', ls='--', lw=1.5)

    g.fig.subplots_adjust(hspace=-0.3)

    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)

    plt.setp(ax.get_xticklabels(), fontsize=15, fontweight='bold')
    plt.xlabel('Jaccard index', fontweight='bold', fontsize=15)
    g.fig.suptitle('Within and between-clusters distributions of pairwise Jaccard scores',
                   fontsize=20,
                   fontweight=20)
    
    # Save figure
    directory = './../figures/figure6'
    if not osp.exists(directory):
        os.makedirs(directory)

    g.savefig(directory + '/figure6.pdf', dpi=500, bbox_inches='tight',
              pad_inches=0.5)
    plt.show()


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
        np.savez('./../output/mean' + str(modality) + '_PAclustering_cluster' + str(cluster_index) + '_' + hemisphere,
                 list=mean)


if __name__ == '__main__':
    path = './../data'

    # Clustering assignment
    with open('./../list_subj.txt') as fp:
        subjects = fp.read().split("\n")
    list_of_subs = np.array(subjects[0:len(subjects) - 1])

    # Cluster assignment
    cluster_assignment = np.load(
        './../output/clusters_individualIndeces_PA.npz')['list']

    clusters = {}
    for i in np.unique(cluster_assignment):
        clusters['cluster_' +
                 str(i + 1)] = list(list_of_subs[cluster_assignment == i])

    # ROI
    label_primary_visual_areas = ['ROI']
    final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
        label_primary_visual_areas)

    modalities = ['polarAngle', 'eccentricity', 'meanbold']

    for modality in modalities:
        for i in range(1, 7):
            averageMaps(path, clusters['cluster_' + str(i)],
                        final_mask_L, final_mask_R, modality, i)
