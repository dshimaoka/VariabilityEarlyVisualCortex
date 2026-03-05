# VariabilityEarlyVisualCortex
Forked from https://github.com/felenitaribeiro/VariabilityEarlyVisualCortex for elastic net simulation

## Main scripts

1, data_formatting_all.m

* Input: (avg) S1200_7T_Retinotopy181.L.midthickness_MSMAll.32k_fs_LR.surf.gii or (individual) subject_id{ii} '.lh.midthickness.32k_fs_LR.surf.gii'

* Output: ['Geom_' subject_id{sid} '.stl'] & ['mid_pos_L_' subject_id{ii}] 


2, export_stl_all.py

* Input: original stl created in data_formatting_all.m
  
* Output: stl after applying a hc laplacian filter ['Geom_' subject_id{sid} '_hclaplacian.stl']


3, export_geometry_all.py

* Input: mid_pos_L_(subjectID).mat (output of data_format_individual.m)

* Output: geometry_retinotopy_(subjectID).mat, containing
grid_x, grid_y, grid_PA, grid_ecc, grid_azimuth, grid_altitude, vfs, final_mask_L, final_mask_L_d, final_mask_L_d_idx, final_mask_L_idx


4, defineArealBorders_all.m

* Input: vfs, grid_azimuth','grid_altitude','grid_ecc from geometry_retinotopy_(subjectID).mat

* Output: 'areaMatrix',"connectedPixels",'vfs_th',"vfs_f", saved in arealBorders_ (subject_id).mat


5, compute_minimal_path_femesh_all.m

* Input: 'array_3d','grid_x','grid_y', 'grid_curv' from geometry_retinotopy_(subject_id).mat & 'areaMatrix' from arealBorder_(subject_id).mat

* Output: 'distance4D','distance2D','xy2node','surfaceNodes','distance4D_euc','distance2D_euc' from minimal_path_(type)_hmax2_(subject_id).mat


6, elastic_sweep_ds_slurm_all.py

* Input: arealBorders_ (subject_id).mat, geometry_retinotopy_(subjectID).mat,

* Output: retinotopy of simulation and original data in polar and cartesian coordinates


## Files created in each subject
Data related to visual area borders

  *  geometry_retinotopy_(subjectID).mat: empirical retinotopy on a flattened cortex (export_geometry_all.py)
  *  arealBorder_(subjectID).mat ()
  *  areaLabels_(subjectID).png: a figure showing the borders of visual areas from V1 to V4 (defineArealBorders_all.m)
  *  areaSegmentation_(subjectID).png: a figure representing processes leading to defining the areal borders (defineArealBorders_all.m)
  *  original_retinotopy_cartesian_(subjectID).png: empirical retinotopy map represented as the cartesian coordinate (elastic_sweep_ds_slurm_all.py)
  *  original_retinotopy_polar_(subjectID).png: empirical retinotopy map represented as the polar coordinate (elastic_sweep_ds_slurm_all.py)

Data related to individual brain geometry

  *  Geom_(subjectID).stl, mid_pos_L_(subjectID).mat: 3D brain geometry data, reformatted from (subjectID).lh.midthickness.32k_fs_LR.surf.gii (data_formatting_all.m)
  *  Geom_(subjectID)_hclaplacian.stl: stl after applying a hc laplacian filter (export_stl_all.py)

Data related to minimal path length in 3D volume

  *   minimal_path_midthickness_hmax2_(subjectID).mat: the minimal path length in a 3D volume between any two locations on the cortical surface (compute_minimal_path_femesh_all.m)
  *   minimal_path_midthickness_hmax2_serie_(subjectID).png: visualization of minimal path_midthickness_hmax2_(subjectID).mat, along with the distances of two control cases (compute_minimal_path_femesh_all.m)
  *   surface_minimal_path_hmax2_(subjectID).png: a figure showing 
        (left) gyrification of individual brain, 
        (middle) curvature values, represented on a flattened surface and 
        (right) shortest path in the 3D volume against one cortical location (red dot), represented on a flattened surface

Data related to elastic net simulation (elastic_sweep_ds_slurm_all.py)

 *   simulated_retinotopy_cartesian(or polar)_(target brain region)_(subjectID)_b1_(b1 parameter value x 1000)_b2_(b2 parameter value x 1000).png: simulated retinotopy map, represented as the cartesian or polar coordinate, using the minimal path length in 3D volume (elastic_sweep_ds_slurm_all.py)
 *   simulated_retinotopy_cartesian(or polar)_(target brain region)_(subjectID)_b1_(b1 parameter value x 1000)_b2_(b2 parameter value x 1000)_flat.png: simulated retinotopy map, using the Euclidean distance on flattened surface (elastic_sweep_ds_slurm_all.py)
 *   summary_(subjectID)(target brain region)_(subjectID)_b1_(b1 parameter value x 1000)_b2_(b2 parameter value x 1000).mat: containing simulated retinotopy maps using (elastic_sweep_ds_slurm_all.py)
 *   summary_correlation_(subjectID).png/.mat: a summary of correlation between empirical and simulated retinotopy in (b1,b2) parameter space (elastic_sweep_ds_slurm_all.py)


