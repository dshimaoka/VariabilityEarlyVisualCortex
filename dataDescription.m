%data produced by data_formatting.m: 
% tri_faces_L: 64980x3 faces coordinates (x,y,z) of a standard brain
% mid_pos_L: 32492x3 vertices coordinates (x,y,z) of a standard brain

%% show 3D surface of the standard brain
mid_gifti_L=gifti('S1200_7T_Retinotopy181.L.midthickness_MSMAll.32k_fs_LR.surf.gii');
patch('Vertices',mid_gifti_L.vertices,'Faces',mid_gifti_L.faces, 'EdgeColor','interp','FaceColor','g');
view(3)
axis vis3d

%% curvature on the spherical brain 
load('cifti_curvature_all.mat'); %originally S1200_7T_Retinotopy181.All.curvature_MSMAll.32k_fs_LR.dscalar.nii
%cifti_curvature.pos: 64984x3 faces of a sphere of a standard brain
%cifti_curvature.tri: 129960x3 vertices of a sphere of a standard brain
patch('Vertices',cifti_curvature.pos,'Faces',cifti_curvature.tri,'EdgeColor','none',...
    'FaceVertexCData',cifti_curvature.x146735_curvature,'FaceColor','interp');
view(3)
axis vis3d

%% polar angle on the spherical brain
load('cifti_polarAngle_all.mat')
patch('Vertices',cifti_polarAngle.pos,'Faces',cifti_polarAngle.tri,'EdgeColor','none',...
    'FaceVertexCData',cifti_polarAngle.x146735_fit1_polarangle_msmall,'FaceColor','interp');
colormap("hsv");
view(3)
axis vis3d

%% conversion
%original 3D <> inflated sphere <> flattend 2D