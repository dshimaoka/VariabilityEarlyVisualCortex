%% visualize path in 3D space between two locations identified by elastic_sweep_ds_slurm_vs.py

subject_id = {'114823'};%setxor(getSubjectId, {'114823','157336','585256','581450','725751'});
saveDir = '/mnt/dshi0006_market/VariabilityEarlyVisualCortex/';

type = 'midthickness';%'white' %cannot generateMesh with 'pial'
hmax = 2; %1: fine but too slow, 3: too coarse
tgt = 'V+D';
b1 = 0.32;
b2 = 0.02;

minscheme = 'volume';%'surface';%
sid = 1;

load(fullfile(saveDir, subject_id{sid}, ['geometry_retinotopy_'  subject_id{sid}   '.mat']),...
    'array_3d','grid_x','grid_y', 'grid_curv');%,'mask');
[x,y,z,xaxis, yaxis, withinMask] = getXYZ(grid_x, grid_y, array_3d);

%% elastic net simulation result
suffix = [tgt '_'  subject_id{sid} '_b1_' num2str(1e3*b1) '_b2_' num2str(1e3*b2)];
saveName_enet = fullfile(saveDir, subject_id{sid},...
    ['summary_' suffix +'_vs.mat']);

load(fullfile(saveDir, subject_id{sid}, ['minimal_path_' type '_hmax' ...
    num2str(hmax) '_' subject_id{sid} '_s.mat']),'Face_s','Vertex_s');
switch minscheme
    %% load graph
    case 'volume'
        saveName_graph = fullfile(saveDir, subject_id{sid}, ['minimal_path_' type '_hmax' ...
            num2str(hmax) '_' subject_id{sid} '_v.mat']);
        load(saveName_graph,'Face_v','Vertex_v','G_v');
        G = G_v;
        Face = Face_v;
        Vertex = Vertex_v;

        load(saveName_enet, 'result2d_v','reg_final4d_v');
        result2d = result2d_v;
        reg_final4d = reg_final4d_v;
    
    case 'surface'
        saveName_graph = fullfile(saveDir, subject_id{sid}, ['minimal_path_' type '_hmax' ...
            num2str(hmax) '_' subject_id{sid} '_s.mat']);
        load(saveName_graph,'Face_s','Vertex_s','G_s');
        G = G_s;
        Face = Face_s;
        Vertex = Vertex_s;

        load(saveName_enet, 'result2d_s','reg_final4d_s');
        result2d = result2d_s;
        reg_final4d = reg_final4d_s;
end
%% compute path_node
tic
[distance4D, path_node4D, ~, xy2node] ...
    = getDistance4D(G, grid_x, grid_y, array_3d);
t3 = toc %~30s


%% visualization

sxi = 29;%31;
syi = 71;
mask = ~isnan(result2d(:,:,1))';
A = squeeze((reg_final4d(sxi,syi,:,:)))';%squeeze(log(1./reg_final4d(sxi,syi,:,:)))';
A(syi,sxi)=0; %exclude recurrent connectivity
fig1 = figure;
ax(1)=subplot(1,3,1); imagesc(xaxis, yaxis, squeeze(result2d(:,:,1))','alphadata',mask); hold on; plot(xaxis(sxi), yaxis(syi),'r*'); axis equal tight xy; title('simulated azimuth');
ax(2)=subplot(1,3,2); imagesc(xaxis, yaxis, squeeze(result2d(:,:,2))','alphadata',mask); hold on; plot(xaxis(sxi), yaxis(syi),'r*'); axis equal tight xy; title('simulated elevation');
ax(3)=subplot(1,3,3); imagesc(xaxis, yaxis, A,'alphadata',mask); hold on; plot(xaxis(sxi), yaxis(syi),'r*'); axis equal tight xy; title('connectivity strength');
linkaxes(ax);
screen2png([saveName_graph(1:end-4) suffix], fig1);
close(fig1);

%% obtain target pixel
%[~, idx] = max(A(:));
[~, idx] = sort(A(:),'descend');
idx(isnan(A(idx)))=[];
N = 5; %number of target nodes
[tyi, txi] = ind2sub(size(A), idx(1:N));
fig2 = examinePathNode(Vertex, distance4D, path_node4D, xaxis, yaxis, ...
    xy2node, sxi*ones(N,1), syi*ones(N,1), txi, tyi, Face_s, Vertex_s, 0.3);
screen2png([saveName_graph(1:end-4) suffix '_pathNode'], fig2);
close(fig2);

