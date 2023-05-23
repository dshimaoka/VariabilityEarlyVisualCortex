clear all

addpath(genpath('.'))

%Connectivity of triangular faces
mid_gifti_L=gifti('S1200_7T_Retinotopy181.L.midthickness_MSMAll.32k_fs_LR.surf.gii');
mid_gifti_R=gifti('S1200_7T_Retinotopy181.R.midthickness_MSMAll.32k_fs_LR.surf.gii');

tri_faces_L=mid_gifti_L.faces;
save('tri_faces_L','tri_faces_L')
tri_faces_R=mid_gifti_R.faces;
save('tri_faces_R','tri_faces_R');

mid_pos_L=mid_gifti_L.vertices;
save('mid_pos_L','mid_pos_L');
mid_pos_R=mid_gifti_R.vertices;
save('mid_pos_R','mid_pos_R');

%Load cifti files and get measures for each cortical vertex, where
%the brain structure labels are so that 1=Left; 2=Right; >2=subcortical structures

%Polar Angle values
cifti_polarAngle=ft_read_cifti('S1200_7T_Retinotopy181.All.Fit1_PolarAngle_MSMAll.32k_fs_LR.dscalar.nii');
save('cifti_polarAngle_all','cifti_polarAngle')

cifti_polarAngle=ft_read_cifti('S1200_7T_Retinotopy181.All.Fit2_PolarAngle_MSMAll.32k_fs_LR.dscalar.nii');
save('cifti_polarAngle_fit2_all','cifti_polarAngle')

cifti_polarAngle=ft_read_cifti('S1200_7T_Retinotopy181.All.Fit3_PolarAngle_MSMAll.32k_fs_LR.dscalar.nii');
save('cifti_polarAngle_fit3_all','cifti_polarAngle')


%Eccentricity values
cifti_eccentricity=ft_read_cifti('S1200_7T_Retinotopy181.All.Fit1_Eccentricity_MSMAll.32k_fs_LR.dscalar.nii');
save('cifti_eccentricity_all','cifti_eccentricity')

cifti_eccentricity=ft_read_cifti('S1200_7T_Retinotopy181.All.Fit2_Eccentricity_MSMAll.32k_fs_LR.dscalar.nii');
save('cifti_eccentricity_fit2_all','cifti_eccentricity')

cifti_eccentricity=ft_read_cifti('S1200_7T_Retinotopy181.All.Fit3_Eccentricity_MSMAll.32k_fs_LR.dscalar.nii');
save('cifti_eccentricity_fit3_all','cifti_eccentricity')

%Mean bold signal
cifti_meanbold=ft_read_cifti('S1200_7T_Retinotopy181.All.Fit1_MeanBOLD_MSMAll.32k_fs_LR.dscalar.nii');
save('cifti_meanbold_all','cifti_meanbold')

%Curvature
cifti_curvature=ft_read_cifti('S1200_7T_Retinotopy181.All.curvature_MSMAll.32k_fs_LR.dscalar.nii');
save('cifti_curvature_all','cifti_curvature')

exit