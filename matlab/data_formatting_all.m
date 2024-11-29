% created from data_formatting.m

clear all

%% extract subject ids
subject_id = getSubjectId;

%subject_id = {'avg','114823','157336','585256','114823','581450','725751'};
% saveDir = '/home/daisuke/Documents/git/VariabilityEarlyVisualCortex/data/';
saveDir = '/mnt/dshi0006_market/VariabilityEarlyVisualCortex/';

addpath(genpath('.'))

for ii = 1:numel(subject_id)
    % %Connectivity of triangular faces
    if strcmp(subject_id{ii}, 'avg')
        loadDir = '/home/daisuke/Documents/git/VariabilityEarlyVisualCortex/data/';
        loadName = fullfile(loadDir, 'S1200_7T_Retinotopy181.L.midthickness_MSMAll.32k_fs_LR.surf.gii');
    else
        loadDir = '/mnt/dshi0006_market/HCPData/';
        loadName = fullfile(loadDir,subject_id{ii},'surf',[subject_id{ii} '.lh.midthickness.32k_fs_LR.surf.gii']);
        % NOT this one:
        % loadName2 = fullfile(loadDir,subject_id{ii},'MNINonLinear','fsaverage_LR32k',[subject_id{ii} '.L.white_MSMAll.32k_fs_LR.surf.gii']);
    end

    mid_gifti_L=gifti(loadName);
    %    faces: [64980×3 int32]
    %      mat: [4×4 double]
    % vertices: [32492×3 single]

    tri_faces_L=mid_gifti_L.faces;
    %save(fullfile(saveDir, ['tri_faces_L_' subject_id{ii}]), 'tri_faces_L'); %may not need to save?

    mid_pos_L=mid_gifti_L.vertices;
    if ~exist(fullfile(saveDir, subject_id{ii}), 'dir')
        mkdir(fullfile(saveDir, subject_id{ii}));
    end
    save(fullfile(saveDir, subject_id{ii},['mid_pos_L_' subject_id{ii}]), 'mid_pos_L'); %USED IN export_geometry_individual

    stlwrite(fullfile(saveDir,subject_id{ii}, ['Geom_'  subject_id{ii} '.stl']), tri_faces_L, mid_pos_L)
    clear mid_gifti_L


    %% curvature 
    % loadName = fullfile(loadDir,subject_id{ii},'surf',[subject_id{ii} '.curvature-midthickness.lh.32k_fs_LR.func.gii']);%not existing    
    % curvature_gifti_L=gifti(loadName);
    % % cdata: [32492×1 single]
    % curvature = curvature_gifti_L.cdata;
    % 
    % save(fullfile(saveDir, subject_id{ii},['gifti_curvature_L_' subject_id{ii}]), 'curvature');
    % clear curvature_gifti_L
end