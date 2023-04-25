import os.path as osp
import sys
import numpy as np
import os.path as osp
import sys
import time
import matplotlib.pyplot as plt
import scipy
import nibabel as nib
import scipy
from scipy.interpolate import griddata

sys.path.append('..')


from functions.def_ROIs_EarlyVisualAreas import roi
from functions.def_ROIs_DorsalEarlyVisualCortex import roi as ROI
from functions.individual_variability import grab_data

def PA_gradients(subject_id, path, plot_type = 'streamplot', binarize = False, save = False, save_path = None):
    """
    Plot the polar angle gradient for the dorsal portion of the early visual cortex.
    Parameters
    ----------
    subject_id : int
        Subject ID.
    path : str
        Path to the data.
    plot_type : str, optional
        Type of plot. The default is 'streamplot'.
    binarize : bool, optional
        Binarize the polar angle map. The default is False.
    save : bool, optional
        Save the figure. The default is False.
    save_path : str, optional
        Path to save the figure. The default is None.
    Returns
    -------
    Plot of the polar angle gradient.   
    """

    
    # Early visual cortex
    label_primary_visual_areas = ['ROI']
    final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
        label_primary_visual_areas)

    # Dorsal portion
    label_primary_visual_areas = ['ROI']
    final_mask_L_dorsal, final_mask_R_dorsal, index_L_mask_dorsal, index_R_mask_dorsal = ROI(
        label_primary_visual_areas)

    number_cortical_nodes = int(64984)
    number_hemi_nodes = int(number_cortical_nodes / 2)

    # Loading the flat surface
    flat_surf = nib.load(osp.join(path,'S1200_7T_Retinotopy181.L.sphere.32k_fs_LR.surf.gii'))
    flat_surf_pos = flat_surf.agg_data('pointset')[final_mask_L*final_mask_L_dorsal==1]
    coord_plane = np.array(flat_surf_pos.T[0:3,].T).astype(int)
    new_coord_plane = np.matmul([[0, 0, 1],[0, 1, 0]], coord_plane.T).T + 100
    
    # Loading the polar angle values
    z_values = np.zeros((number_hemi_nodes, 1))
    data = scipy.io.loadmat(osp.join(path, 'cifti_polarAngle_all.mat'))[
            'cifti_polarAngle']
    z_values[final_mask_L*final_mask_L_dorsal == 1] = np.reshape(
                data['x' + str(subject_id) + '_fit1_polarangle_msmall'][0][0][
                    0:number_hemi_nodes].reshape(
                        (number_hemi_nodes))[final_mask_L*final_mask_L_dorsal == 1], (-1, 1))
    z_values[final_mask_L*final_mask_L_dorsal != 1] = 0


    # Interpolating the polar angle values
    grid_x, grid_y = np.mgrid[40:120, 0:60]
    grid_z0 = griddata(new_coord_plane, z_values[final_mask_L*final_mask_L_dorsal == 1], (grid_x, grid_y), method='linear')

    # Determining the gradient
    dx, dy = np.gradient(grid_z0[:,:,0])

    # Binarizing the polar angle map
    if binarize == True:
        z_values[(z_values >= 0) & (z_values <= 45)] = 0
        z_values[(z_values > 45) & (z_values <= 180)]= 90
        z_values[(z_values >= 315) & (z_values <= 360)] = 360
        z_values[(z_values > 180) & (z_values < 315)] = 270
        z_values[final_mask_L*final_mask_L_dorsal != 1] = 0
        grid_z0 = griddata(new_coord_plane, z_values[final_mask_L*final_mask_L_dorsal == 1], (grid_x, grid_y), method='linear')

    # Sampling the gradient
    X = grid_x[::2, ::2]
    Y = grid_y[::2, ::2]
    U = dx[::2, ::2]
    V = dy[::2, ::2]

    # Plotting the gradient
    plt.imshow(grid_z0[:,:,0].T, extent=[40,120, 0,60], origin='lower', cmap='gist_rainbow_r', vmax = 361)
    if plot_type == 'streamplot':
        plt.streamplot(X.T, Y.T, U.T, V.T,density = 2, linewidth=1, arrowsize=1, arrowstyle='->', color='k')
    elif plot_type == 'quiver':
        plt.quiver(X, Y, U, V,scale = 10, units = 'x')

    if save == True:
        plt.savefig(osp.join(save_path, 'PA_gradients_' + subject_id  + '.png'))
    return plt.show()

if __name__ == '__main__':
    subject_id = 100610
    path = './../data'
    PA_gradients(subject_id, path, plot_type = 'streamplot')