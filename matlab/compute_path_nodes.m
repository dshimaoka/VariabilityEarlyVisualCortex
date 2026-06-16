subject_id = setxor(getSubjectId, {'114823','157336','585256','581450','725751'});
%114823: meshing failed for hmax of 2 and hmin of 1

saveDir = '/mnt/dshi0006_market/VariabilityEarlyVisualCortex/';

type = 'midthickness';%'white' %cannot generateMesh with 'pial'
hmax = 2; %1: fine but too slow, 3: too coarse


sid = 1;

%% load data
loadDir = '/mnt/dshi0006_market/HCPData/';
loadName = fullfile(loadDir,subject_id{sid},'surf',[subject_id{sid} '.lh.midthickness.32k_fs_LR.surf.gii']);
mid_gifti_L=gifti(loadName); %from data_formatting_all.m

load(fullfile(saveDir, subject_id{sid}, ['minimal_path_' type '_hmax' num2str(hmax) '_' subject_id{sid}]),...
    'surfaceNodes','distance4D','xy2node'); %from compute_minimal_path_femesh_all.m
load(fullfile(saveDir, subject_id{sid}, ['geometry_retinotopy_'  subject_id{sid}   '.mat']),...
    'array_3d','grid_x','grid_y');%, 'grid_curv');%,'mask');
load(fullfile(saveDir,subject_id{sid},['arealBorder_' subject_id{sid}]),...
    'areaMatrix');% from defineArealBorders_individual.m

[surfaceNodes_unq, idx_unq] = unique(surfaceNodes);

%% 1. import entire brain volume
tic
model = createpde(1);
importGeometry(model, fullfile(saveDir, subject_id{sid}, ['Geom_'  subject_id{sid} '_hclaplacian.stl'])); %"BracketTwoHoles.stl");%
%pdegplot(model)
generateMesh(model,"Hmax",hmax);%,"geometricOrder","linear","Hmin",0.2*mm); %determines coarseness of the mesh

t1=toc %20s

tic;
nodes = model.Mesh.Nodes';
elements = model.Mesh.Elements';

% Extract vertices (nodes)
vertices = nodes;

% Initialize an empty array to store faces
faces = [];

% Loop through all elements to extract faces
parfor i = 1:size(elements, 1)
    elem = elements(i, :);

    % Define faces of the current element
    elem_faces = [elem([1, 2, 3]);  % Face 1
        elem([1, 2, 4]);  % Face 2
        elem([1, 3, 4]);  % Face 3
        elem([2, 3, 4])]; % Face 4

    % Add faces to the faces list
    faces = [faces; elem_faces];
end

% Ensure unique faces (remove duplicate faces if any)
unique_faces = unique(sort(faces, 2), 'rows'); %UNUSED??
t2=toc %~30s


%% 3. convert FEMesh to graph
tic
% Assuming your FEMesh object has nodes and elements properties
nodes = model.Mesh.Nodes';
elements = model.Mesh.Elements';

% Create an adjacency matrix based on element connectivity
num_nodes = size(nodes, 1);

% Create a graph object from edges and weights
elem_cell = cell(size(elements,1),1);
% Convert the matrix into cell array
for i = 1:numel(elem_cell)
    elem_cell{i} = elements(i,:);
end
[index,  sourcenode, tgtnode, distance] = cellfun(@(x)computeAdjacency(x, nodes, [num_nodes num_nodes]), elem_cell,'UniformOutput',false);
index_all = [index{:}]; index_all = index_all(:);
distance_all = [distance{:}]; distance_all = distance_all(:);
sourcenode_all = [sourcenode{:}]; sourcenode_all = sourcenode_all(:);
tgtnode_all = [tgtnode{:}]; tgtnode_all = tgtnode_all(:);

G = graph([sourcenode_all tgtnode_all], [tgtnode_all sourcenode_all], [distance_all distance_all]);


xaxis = squeeze(grid_x(:,1))';
yaxis = squeeze(grid_y(1,:));
x = reshape(array_3d(:,:,1),numel(xaxis)*numel(yaxis),1);
y = reshape(array_3d(:,:,2),numel(xaxis)*numel(yaxis),1);
z = reshape(array_3d(:,:,3),numel(xaxis)*numel(yaxis),1);
%omit pixels outside the mask
withinMask = find(~isnan(x));
x = x(withinMask);
y = y(withinMask);
z = z(withinMask);

%% compute shortest path between two points in the volume
% distance4D = nan(numel(yaxis), numel(xaxis), numel(yaxis), numel(xaxis));
path_node4D = cell(numel(yaxis), numel(xaxis), numel(yaxis), numel(xaxis));
tic;
%TODO find surface nodes corresponding to the visual cortex
[ty,tx] = ind2sub([numel(yaxis) numel(xaxis)], withinMask);
for snode=1:numel(surfaceNodes_unq)
    disp(snode)
    [path_nodes, distance_all_unq] = shortestpathtree(G, surfaceNodes_unq(snode), ...
        surfaceNodes_unq,'OutputForm','cell');

    %distance_all = nan(1,numel(surfaceNodes));
    path_node_all = cell(1,numel(surfaceNodes));
    for tt = 1:numel(surfaceNodes_unq)
        idx = find(surfaceNodes == surfaceNodes_unq(tt));
        % distance_all(idx) = distance_all_unq(tt);
        path_node_all(idx) = path_nodes(tt);
    end

    [snodes] = find(surfaceNodes == surfaceNodes_unq(snode));

    for ss = 1:numel(snodes)
        [sy,sx] = ind2sub([numel(yaxis) numel(xaxis)], withinMask(snodes(ss)));
        for tnode=1:numel(surfaceNodes)
            %distance4D(sy,sx,ty(tnode),tx(tnode)) = distance_all(tnode);
            path_node4D{sy,sx,ty(tnode),tx(tnode)} = path_node_all{tnode};
        end
    end
end
%distance2D = reshape(distance4D, numel(yaxis)*numel(xaxis), numel(yaxis)*numel(xaxis));
path_node2D = reshape(path_node4D,numel(yaxis)*numel(xaxis), numel(yaxis)*numel(xaxis));
t=toc


save(fullfile(saveDir, subject_id{sid}, ['minimal_path_' type '_hmax' ...
    num2str(hmax) '_' subject_id{sid}]),'path_node4D','-append','-v7.3');


sxi = 40; syi = 60; %source pixel position on flattend map
txi = [15:10:55]; tyi = 70*ones(1,numel(txi)); %target pixel position on flattend map

vidx_s = xy2node(syi,sxi);%    flat2vert(syi,sxi);
vidx_t = [];
for ii = 1:numel(txi)
    vidx_t(ii) = xy2node(tyi(ii),txi(ii)); %flat2vert(tyi(ii),txi(ii));
end

figure;
subplot(141); %distance on flattened map from (sx,sy)
imagesc(xaxis, yaxis, squeeze(distance4D(syi,sxi,:,:)));hold on;
scatter(xaxis(sxi), yaxis(syi), 20, 'filled', 'MarkerFaceColor', 'r');
for ii = 1:numel(txi)
    scatter(xaxis(txi(ii)), yaxis(tyi(ii)), 20, 'filled', 'MarkerFaceColor', 'g');
    lc = [ii/numel(txi) 0 0];
    line([xaxis(sxi) xaxis(txi(ii))], [yaxis(syi) yaxis(tyi(ii))],'color',lc);
end
xlabel('x'); ylabel('y');
title('shortest path distance');
colorbar;
axis equal tight xy; grid on;

for vv = 1:3
    ax(vv) = subplot(1,4,vv+1); %minimal path between (sx,sy) and (tx,ty)
    trisurf(mid_gifti_L.faces, mid_gifti_L.vertices(:,1),...
        mid_gifti_L.vertices(:,2),mid_gifti_L.vertices(:,3),'facealpha',.1, 'edgealpha',.1);
    hold on;
    % trisurf(unique_faces, vertices(:,1), vertices(:,2), vertices(:,3), 'FaceColor', ...
    %     [.25 .25 .25], 'EdgeColor', 'none','facealpha',.3); hold on;
    for ii = 1:numel(txi)
        scatter3(vertices(vidx_s, 1), vertices(vidx_s, 2), vertices(vidx_s, 3), 20, 'r', 'filled');
        scatter3(vertices(vidx_t(ii), 1), vertices(vidx_t(ii), 2), vertices(vidx_t(ii), 3), 20, 'green', 'filled');
    end
    for ii = 1:numel(txi)
        lc = [ii/numel(txi) 0 0];
        plot3(vertices(path_node4D{syi,sxi,tyi(ii),txi(ii)}, 1), ...
            vertices(path_node4D{syi,sxi,tyi(ii),txi(ii)}, 2), ...
            vertices(path_node4D{syi,sxi,tyi(ii),txi(ii)}, 3), '-', 'color',lc,'LineWidth', 2);
    end
    set(gca,'view',[30 10]);
    xlabel('x'); ylabel('y'); zlabel('z');
    axis equal tight;
    ylim([-100 -30]);
    hold off;

    switch vv
        case 1
            set(ax(vv),'view',[0 90]);
        case 2
            set(ax(vv),'view',[90 0]);
        case 3
            set(ax(vv),'view',[0 0]);
    end
end


% %% conversion from position on flattend map to vertex index
% flat2vert = nan(numel(yaxis), numel(xaxis));
% for snode = 1:numel(surfaceNodes_unq)
%     [snodes] = find(surfaceNodes == surfaceNodes_unq(snode));
%     % if numel(snodes)>1
%     %     disp(num2str(numel(snodes)));
%     % end
%     for ss = 1:numel(snodes)
%         [sy,sx] = ind2sub([numel(yaxis) numel(xaxis)], withinMask(snodes(ss)));
%         flat2vert(sy,sx) = surfaceNodes_unq(snode);%snode;
%     end
% end
% 
% 
% %% test visualizing path_nodes
% sx = 50; sy = 60; %source pixel position on flattend map
% tx = [30:5:55]; ty = 70*ones(1,numel(tx)); %target pixel position on flattend map
% vidx_s = flat2vert(sy,sx);
% vidx_t = [];
% for ii = 1:numel(tx)
%     vidx_t(ii) = flat2vert(ty(ii),tx(ii));
% end
% 
% figure;
% subplot(131); %distance on flattened map from (sx,sy)
% imagesc(xaxis, yaxis, squeeze(distance4D(sy,sx,:,:)));hold on;
% scatter(xaxis(sx), yaxis(sy), 20, 'filled', 'MarkerFaceColor', 'r');
% for ii = 1:numel(tx)
%     scatter(xaxis(tx(ii)), yaxis(ty(ii)), 20, 'filled', 'MarkerFaceColor', 'g');
%     lc = [ii/numel(tx) 0 0];
%     line([xaxis(sx) xaxis(tx(ii))], [yaxis(sy) yaxis(ty(ii))],'color',lc);
% end
% xlabel('x'); ylabel('y');
% title('shortest path distance');
% colorbar;
% axis equal tight xy; grid on;
% %clim([0 20])
% 
% subplot(1,3,2:3); %minimal path between (sx,sy) and (tx,ty)
% trisurf(mid_gifti_L.faces, mid_gifti_L.vertices(:,1),...
%     mid_gifti_L.vertices(:,2),mid_gifti_L.vertices(:,3),'facealpha',.2, 'edgealpha',.1);
% hold on;
% % trisurf(unique_faces, vertices(:,1), vertices(:,2), vertices(:,3), 'FaceColor', ...
% %     [.25 .25 .25], 'EdgeColor', 'none','facealpha',.3); hold on;
% for ii = 1:numel(tx)
%     scatter3(vertices(vidx_s, 1), vertices(vidx_s, 2), vertices(vidx_s, 3), 20, 'red', 'filled');
%     scatter3(vertices(vidx_t(ii), 1), vertices(vidx_t(ii), 2), vertices(vidx_t(ii), 3), 20, 'green', 'filled');
% end
% for ii = 1:numel(tx)
%     lc = [ii/numel(tx) 0 0];
%     plot3(vertices(path_node4D{sy,sx,ty(ii),tx(ii)}, 1), ...
%         vertices(path_node4D{sy,sx,ty(ii),tx(ii)}, 2), ...
%         vertices(path_node4D{sy,sx,ty(ii),tx(ii)}, 3), '-', 'color',lc,'LineWidth', 2);
% end
% set(gca,'view',[30 10]);
% xlabel('x'); ylabel('y');
% ylim([-100 0]);
% axis equal tight;
% hold off;




