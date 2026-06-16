subject_id = setxor(getSubjectId, {'114823','157336','585256','581450','725751'});
%114823: meshing failed for hmax of 2 and hmin of 1

saveDir = '/mnt/dshi0006_market/VariabilityEarlyVisualCortex/';

type = 'midthickness';%'white' %cannot generateMesh with 'pial'
% hmax = 2; %1: fine but too slow, 3: too coarse


sid = 1;

%% load data
loadDir = '/mnt/dshi0006_market/HCPData/';
loadName = fullfile(loadDir,subject_id{sid},'surf',[subject_id{sid} '.lh.midthickness.32k_fs_LR.surf.gii']);
mid_gifti_L=gifti(loadName); %from data_formatting_all.m

% load(fullfile(saveDir, subject_id{sid}, ['minimal_path_' type '_hmax' num2str(hmax) '_' subject_id{sid}]),...
%     'surfaceNodes_surf','distance4D_surf'); %from compute_minimal_path_femesh_all.m
load(fullfile(saveDir, subject_id{sid}, ['geometry_retinotopy_'  subject_id{sid}   '.mat']),...
    'array_3d','grid_x','grid_y');%, 'grid_curv');%,'mask');
load(fullfile(saveDir,subject_id{sid},['arealBorder_' subject_id{sid}]),...
    'areaMatrix');% from defineArealBorders_individual.m

%% 3': convert brain surface (.gifti) to graph
edges_surf = [
    mid_gifti_L.faces(:,[1 2]);
    mid_gifti_L.faces(:,[2 3]);
    mid_gifti_L.faces(:,[3 1]);
];

% Remove duplicate edges
edges_surf = sort(edges_surf,2);
edges_surf = unique(edges_surf,'rows');

% create a weighted graph
v1 = mid_gifti_L.vertices(edges_surf(:,1),:);
v2 = mid_gifti_L.vertices(edges_surf(:,2),:);

weights = sqrt(sum((v1-v2).^2,2));

G_surf = graph(edges_surf(:,1), edges_surf(:,2), weights);
G_surf.Nodes.X = mid_gifti_L.vertices(:,1);
G_surf.Nodes.Y = mid_gifti_L.vertices(:,2);
G_surf.Nodes.Z = mid_gifti_L.vertices(:,3);

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

%% obtain nodes on the surface ... its order is according to order of x ... completely random
surfaceNodes_surf = zeros(numel(x),1);
for idx = 1:numel(x)
    [~,surfaceNodes_surf(idx)] = min(abs(mid_gifti_L.vertices(:,1) - x(idx)).^2 + abs(mid_gifti_L.vertices(:,2) - y(idx)).^2 + abs(mid_gifti_L.vertices(:,3) - z(idx)).^2);
end

[surfaceNodes_surf_unq, idx_unq] = unique(surfaceNodes_surf); 

%% compute shortest path between two points along the surface
distance4D_surf = nan(numel(yaxis), numel(xaxis), numel(yaxis), numel(xaxis));
path_node4D_surf = cell(numel(yaxis), numel(xaxis), numel(yaxis), numel(xaxis));
tic;
[ty,tx] = ind2sub([numel(yaxis) numel(xaxis)], withinMask);

for snode=1:numel(surfaceNodes_surf_unq)
    disp(num2str(snode/numel(surfaceNodes_surf_unq)*100));
    [path_nodes, distance_all_unq] = shortestpathtree(G_surf, ...
        surfaceNodes_surf_unq(snode), surfaceNodes_surf_unq,...
        'OutputForm','cell');

    distance_all = nan(1,numel(surfaceNodes_surf));
    path_node_all = cell(1,numel(surfaceNodes_surf));
    for tt = 1:numel(surfaceNodes_surf_unq)
        idx = find(surfaceNodes_surf == surfaceNodes_surf_unq(tt));
        distance_all(idx) = distance_all_unq(tt);
        path_node_all(idx) = path_nodes(tt);
    end

    [snodes] = find(surfaceNodes_surf == surfaceNodes_surf_unq(snode));

    [ty,tx] = ind2sub([numel(yaxis) numel(xaxis)], withinMask);
    for ss = 1:numel(snodes)
        [sy,sx] = ind2sub([numel(yaxis) numel(xaxis)], withinMask(snodes(ss)));
        for tnode=1:numel(surfaceNodes_surf)
            distance4D_surf(sy,sx,ty(tnode),tx(tnode)) = distance_all(tnode);
            path_node4D_surf{sy,sx,ty(tnode),tx(tnode)} = path_node_all{tnode};
        end
    end
end
distance2D_surf = reshape(distance4D_surf, numel(yaxis)*numel(xaxis), numel(yaxis)*numel(xaxis));
path_node2D_surf = reshape(path_node4D_surf,numel(yaxis)*numel(xaxis), numel(yaxis)*numel(xaxis));



%% test visualizing path_nodes
sxi = 40; syi = 60; %source pixel position on flattend map
txi = [15:10:55]; tyi = 70*ones(1,numel(txi)); %target pixel position on flattend map


%% conversion from position on flattend map to vertex index
vertices = table2array(G_surf.Nodes);

xy2node_surf = nan(numel(yaxis),numel(xaxis));
xy2node_surf(withinMask) = surfaceNodes_surf;

vidx_s = xy2node_surf(syi,sxi);%    flat2vert(syi,sxi);
vidx_t = [];
for ii = 1:numel(txi)
    vidx_t(ii) = xy2node_surf(tyi(ii),txi(ii)); %flat2vert(tyi(ii),txi(ii));
end

figure;
subplot(141); %distance on flattened map from (sx,sy)
imagesc(xaxis, yaxis, squeeze(distance4D_surf(syi,sxi,:,:)));hold on;
scatter(xaxis(sxi), yaxis(syi), 20, 'filled', 'MarkerFaceColor', 'm');
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
        scatter3(vertices(vidx_s, 1), vertices(vidx_s, 2), vertices(vidx_s, 3), 20, 'm', 'filled');
        scatter3(vertices(vidx_t(ii), 1), vertices(vidx_t(ii), 2), vertices(vidx_t(ii), 3), 20, 'green', 'filled');
    end
    for ii = 1:numel(txi)
        lc = [ii/numel(txi) 0 0];
        plot3(vertices(path_node4D_surf{syi,sxi,tyi(ii),txi(ii)}, 1), ...
            vertices(path_node4D_surf{syi,sxi,tyi(ii),txi(ii)}, 2), ...
            vertices(path_node4D_surf{syi,sxi,tyi(ii),txi(ii)}, 3), '-', 'color',lc,'LineWidth', 2);
    end
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


%% save result
save(fullfile(saveDir, subject_id{sid}, ['minimal_path_' type '_surf_' subject_id{sid}]),...
    'distance4D_surf','distance2D_surf','surfaceNodes_surf',...
    'xy2node','path_node4D_surf','-v7.3');%,'path_node2D_surf' 'G_surf'; %SLOW
