import os.path as osp
import sys
import numpy as np
import os.path as osp
import sys
import time
import matplotlib.pyplot as plt
import scipy
import nibabel as nib
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata

sys.path.append('..')


from functions.def_ROIs_EarlyVisualAreas import roi
from functions.def_ROIs_DorsalEarlyVisualCortex import roi as ROI
from functions.individual_variability import grab_data

def PA_gradients(subject_id, path, plot_type = 'streamplot', binarize = False, dorsal_only=True, save = False, save_path = None):
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
    dorsal_only : bool, optional
        Plot only the dorsal portion of the early visual cortex. The default is True.
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
    if dorsal_only ==False:
        final_mask_L_dorsal = np.ones(np.shape(final_mask_L_dorsal))
    # Number of nodes
    number_cortical_nodes = int(64984)
    number_hemi_nodes = int(number_cortical_nodes / 2)

    # Loading the flat surface
    flat_surf = nib.load(osp.join(path,'S1200_7T_Retinotopy181.L.sphere.32k_fs_LR.surf.gii'))
    flat_surf_pos = flat_surf.agg_data('pointset')[final_mask_L*final_mask_L_dorsal==1]
    coord_plane = np.array(flat_surf_pos.T[0:3,].T).astype(int)
    new_coord_plane = np.matmul([[0, 0, 1],[0, 1, 0]], coord_plane.T).T + 100
    
    # Loading polar angle values
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
    if dorsal_only == False:
        grid_x, grid_y = np.mgrid[20:120, 0:100]
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
    if dorsal_only == True:
        plt.imshow(grid_z0[:,:,0].T, extent=[40,120, 0,60], origin='lower', cmap='gist_rainbow_r', vmax = 361)
    if dorsal_only == False:
        plt.imshow(grid_z0[:,:,0].T, extent=[20,120, 0,100], origin='lower', cmap='gist_rainbow_r', vmax = 361)
    if plot_type == 'streamplot':
        plt.streamplot(X.T, Y.T, U.T, V.T,density = 2, linewidth=1, arrowsize=1, arrowstyle='->', color='k')
    elif plot_type == 'quiver':
        plt.quiver(X, Y, U, V,scale = 10, units = 'x')

    if save == True:
        plt.savefig(osp.join(save_path, 'PA_gradients_' + str(subject_id)  + '.png'))
    return plt.show()

def fieldSign(subject_id, path, dorsal_only=True, smoothing = False, save = False, save_path = None):
    """
    Plot the field sign for the dorsal portion of the early visual cortex.
    Parameters
    ----------
    subject_id : int
        Subject ID.
    path : str
        Path to the data.
    dorsal_only : bool, optional
        Plot the dorsal portion only. The default is True.
    smoothing : bool, optional
        Smooth the polar angle interpolation. The default is False.
    save : bool, optional
        Save the figure. The default is False.
    save_path : str, optional
        Path to save the figure. The default is None.
    Returns
    -------
    Plot of the field sign. 
    """

    # Early visual cortex
    label_primary_visual_areas = ['ROI']
    final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
        label_primary_visual_areas)

    # Dorsal portion
    label_primary_visual_areas = ['ROI']
    final_mask_L_dorsal, final_mask_R_dorsal, index_L_mask_dorsal, index_R_mask_dorsal = ROI(
        label_primary_visual_areas)
    if dorsal_only ==False:
        final_mask_L_dorsal = np.ones(np.shape(final_mask_L_dorsal))
    # Number of nodes
    number_cortical_nodes = int(64984)
    number_hemi_nodes = int(number_cortical_nodes / 2)

    # Loading the flat surface
    flat_surf = nib.load(osp.join(path,'S1200_7T_Retinotopy181.L.sphere.32k_fs_LR.surf.gii'))
    flat_surf_pos = flat_surf.agg_data('pointset')[final_mask_L*final_mask_L_dorsal==1]
    coord_plane = np.array(flat_surf_pos.T[0:3,].T).astype(int)
    new_coord_plane = np.matmul([[0, 0, 1],[0, 1, 0]], coord_plane.T).T + 100


    #### Polar angle map ####
    # Loading polar angle values
    z_values_PA = np.zeros((number_hemi_nodes, 1))
    data = scipy.io.loadmat(osp.join(path, 'cifti_polarAngle_all.mat'))[
            'cifti_polarAngle']
    z_values_PA[final_mask_L*final_mask_L_dorsal == 1] = np.reshape(
                data['x' + str(subject_id) + '_fit1_polarangle_msmall'][0][0][
                    0:number_hemi_nodes].reshape(
                        (number_hemi_nodes))[final_mask_L*final_mask_L_dorsal == 1], (-1, 1))
    # Shifting PA values
    sum = z_values_PA < 180
    minus = z_values_PA > 180
    z_values_PA[sum] = z_values_PA[sum] + 180
    z_values_PA[minus] = z_values_PA[minus] - 180

    z_values_PA[final_mask_L*final_mask_L_dorsal != 1] = 0

    # Interpolating the polar angle values
    grid_x, grid_y = np.mgrid[40:120, 0:60]
    if dorsal_only == False:
        grid_x, grid_y = np.mgrid[20:120, 0:100]
    grid_z0_PA = griddata(new_coord_plane, z_values_PA[final_mask_L*final_mask_L_dorsal == 1], (grid_x, grid_y), method='linear')

    # Smoothing the polar angle map
    if smoothing == True:
        grid_z0_PA[:,:,0] = gaussian_filter(grid_z0_PA[:,:,0], sigma=.8)
    
    # Determining the gradient
    dx_PA, dy_PA = np.gradient(grid_z0_PA[:,:,0])

    #### Eccentricity map ####
    # Loading eccentricity values
    z_values_ecc = np.zeros((number_hemi_nodes, 1))
    data = scipy.io.loadmat(osp.join(path, 'cifti_eccentricity_all.mat'))[
            'cifti_eccentricity']
    z_values_ecc[final_mask_L*final_mask_L_dorsal == 1] = np.reshape(
                data['x' + str(subject_id) + '_fit1_eccentricity_msmall'][0][0][
                    0:number_hemi_nodes].reshape(
                        (number_hemi_nodes))[final_mask_L*final_mask_L_dorsal == 1], (-1, 1))
    z_values_ecc[final_mask_L*final_mask_L_dorsal != 1] = 0

    # Interpolating the eccentricity values
    grid_z0_ecc = griddata(new_coord_plane, z_values_ecc[final_mask_L*final_mask_L_dorsal == 1], (grid_x, grid_y), method='linear')

    # Determining the gradient
    dx_ecc, dy_ecc = np.gradient(grid_z0_ecc[:,:,0])

    #### Field sign analysis ####
    # Angle between gradient vectors
    dot_product = dx_PA*dx_ecc + dy_PA*dy_ecc
    modulus_PA = np.sqrt(dx_PA**2 + dy_PA**2)
    modulus_ecc = np.sqrt(dx_ecc**2 + dy_ecc**2)
    theta = np.arccos(dot_product/(modulus_PA*modulus_ecc))
    
    # Cross product
    cross_product = dx_PA*dy_ecc - dy_PA*dx_ecc

    # Binarizing
    theta[cross_product<0] = 2*np.pi - theta[cross_product<0]
    theta[theta>np.pi] = 2*np.pi
    theta[theta<np.pi] = np.pi

    # Plotting the visual field sign
    if dorsal_only == True:
        plt.imshow(theta[:,:].T, extent=[40,120, 0,60], origin='lower', cmap='viridis')
    if dorsal_only == False:
        plt.imshow(theta[:,:].T, extent=[20,120, 0,100], origin='lower', cmap='viridis')
    if save == True:
        if smoothing==False:
            plt.savefig(osp.join(save_path, 'fieldSign_' + str(subject_id)  + '.png'))
        else:
            plt.savefig(osp.join(save_path, 'fieldSign_' + str(subject_id)  + '_smoothed.png'))
    return plt.show()



if __name__ == '__main__':
    subjects = [397760, 926862, 105923, 114823, 198653, 789373] # Figure 7
    # subjects = [585256]
    for subject_id in subjects:
        path = './../data'
        PA_gradients(subject_id, path, plot_type = 'streamplot', dorsal_only = False, save=True, save_path = './../figures')
        fieldSign(subject_id, path, save=True, dorsal_only=False, smoothing=True, save_path = './../figures')