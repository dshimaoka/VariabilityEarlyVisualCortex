% created from data_formatting.m

clear all

subject_id = {'114823','157336','585256','114823','581450','725751'};
saveDir = '/home/daisuke/Documents/git/VariabilityEarlyVisualCortex/data/';

addpath(genpath('.'))

for ii = 1:numel(subject_id)
    % %Connectivity of triangular faces
    loadName = [subject_id{ii} '.lh.midthickness.32k_fs_LR.surf.gii'];
    mid_gifti_L=gifti(loadName);
    %    faces: [64980×3 int32]
    %      mat: [4×4 double]
    % vertices: [32492×3 single]

    tri_faces_L=mid_gifti_L.faces;
    save(fullfile(saveDir, ['tri_faces_L_' subject_id{ii}]), 'tri_faces_L')

    mid_pos_L=mid_gifti_L.vertices;
    save(fullfile(saveDir, ['mid_pos_L_' subject_id{ii}]), 'mid_pos_L');


    %% curvature 
    loadName = [subject_id{ii} '.curvature-midthickness.lh.32k_fs_LR.func.gii'];
    curvature_gifti_L=gifti(loadName);
    % cdata: [32492×1 single]
    curvature = curvature_gifti_L.cdata;

    save(fullfile(saveDir, ['gifti_curvature_L_' subject_id{ii}]), 'curvature');

end