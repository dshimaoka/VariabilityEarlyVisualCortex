%use the same mesh but only its surface part

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
    'surfaceNodes','xy2node'); %from compute_minimal_path_femesh_all.m
load(fullfile(saveDir, subject_id{sid}, ['geometry_retinotopy_'  subject_id{sid}   '.mat']),...
    'array_3d','grid_x','grid_y');%, 'grid_curv');%,'mask');
load(fullfile(saveDir,subject_id{sid},['arealBorder_' subject_id{sid}]),...
    'areaMatrix');% from defineArealBorders_individual.m

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
    %figure;plot3(x,y,z,'.');xlabel('x');ylabel('y');zlabel('z');


    %% 1. import entire brain
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
    unique_faces = unique(sort(faces, 2), 'rows');
    t2=toc %~30s

    %% 3. convert FEMesh to graph
    tic
    % Assuming your FEMesh object has nodes and elements properties
    nodes = model.Mesh.Nodes';
    elements = model.Mesh.Elements';

    % % Create an adjacency matrix based on element connectivity
    % num_nodes = size(nodes, 1);
    % 
    % % Create a graph object from edges and weights
    % elem_cell = cell(size(elements,1),1);
    % % Convert the matrix into cell array
    % for i = 1:numel(elem_cell)
    %     elem_cell{i} = elements(i,:);
    % end
    % [index,  sourcenode, tgtnode, distance] = cellfun(@(x)computeAdjacency(x, nodes, [num_nodes num_nodes]), elem_cell,'UniformOutput',false);
    % index_all = [index{:}]; index_all = index_all(:);
    % distance_all = [distance{:}]; distance_all = distance_all(:);
    % sourcenode_all = [sourcenode{:}]; sourcenode_all = sourcenode_all(:);
    % tgtnode_all = [tgtnode{:}]; tgtnode_all = tgtnode_all(:);
    % 
    % G = graph([sourcenode_all tgtnode_all], [tgtnode_all sourcenode_all], [distance_all distance_all]);
    % 
    % t3 = toc
    % 
    % 
    % %% obtain nodes on the surface ... its order is according to order of x ... completely random
    % surfaceNodes = zeros(numel(x),1);
    % for idx = 1:numel(x)
    %     [~,surfaceNodes(idx)] = min(abs(nodes(:,1) - x(idx)).^2 + abs(nodes(:,2) - y(idx)).^2 + abs(nodes(:,3) - z(idx)).^2);
    % end
    % 
    % [surfaceNodes_unq, idx_unq] = unique(surfaceNodes);
    % 
    % xy2node = nan(numel(yaxis),numel(xaxis));
    % xy2node(withinMask) = surfaceNodes;


%% create surface mesh
tic

T = elements;

surfaceMask = false(size(nodes,1),1);
surfaceMask(surfaceNodes) = true;

theseedges = [];

for k = 1:size(T,1)

    thesenodes = T(k,:);

    thesenodes = thesenodes(surfaceMask(thesenodes));

    if numel(thesenodes) < 2
        continue
    end

    E = nchoosek(thesenodes,2);

    theseedges = [theseedges; E];
end

theseedges = sort(theseedges,2);
theseedges = unique(theseedges,'rows');

newID = zeros(size(nodes,1),1);
newID(surfaceNodes) = 1:numel(surfaceNodes);
edges2 = newID(theseedges);

Gsurf = graph(edges2(:,1), edges2(:,2));
Gsurf.Nodes.X = nodes(surfaceNodes,1);
Gsurf.Nodes.Y = nodes(surfaceNodes,2);
Gsurf.Nodes.Z = nodes(surfaceNodes,3);

surfaceNodes_surf = newID(surfaceNodes);
[surfaceNodes_surf_unq, idx_unq] = unique(surfaceNodes_surf);
xy2node_surf = nan(numel(yaxis),numel(xaxis));
xy2node_surf(withinMask) = surfaceNodes_surf;
vertices_surf = vertices(surfaceNodes,:);


% Assuming your FEMesh object has nodes and elements properties
% nodes = model.Mesh.Nodes';
% elements = model.Mesh.Elements';
% TR = triangulation(elements(:,1:4), nodes);
% F = freeBoundary(TR);
% 
% edges = [F(:,[1 2]);
%          F(:,[2 3]);
%          F(:,[3 1])];
% 
% edges = sort(edges,2);
% edges = unique(edges,'rows');
% 
% L = vecnorm(nodes(edges(:,1),:) - nodes(edges(:,2),:),2,2);
% Gsurf = graph(edges(:,1),edges(:,2),L);
% Gsurf.Nodes.X = nodes()
% 
% xaxis = squeeze(grid_x(:,1))';
% yaxis = squeeze(grid_y(1,:));
% x = reshape(array_3d(:,:,1),numel(xaxis)*numel(yaxis),1);
% y = reshape(array_3d(:,:,2),numel(xaxis)*numel(yaxis),1);
% z = reshape(array_3d(:,:,3),numel(xaxis)*numel(yaxis),1);
% %omit pixels outside the mask
% withinMask = find(~isnan(x));
% x = x(withinMask);
% y = y(withinMask);
% z = z(withinMask);
% 
% surfaceNodes_surf = zeros(numel(x),1);
% for idx = 1:numel(x)
%     [~,surfaceNodes_surf(idx)] = min(abs(Gsurf.Nodes(:,1) - x(idx)).^2 ...
%         + abs(Gsurf.Nodes(:,2) - y(idx)).^2 ...
%         + abs(Gsurf.Nodes(:,3) - z(idx)).^2);
% end
% 
% [surfaceNodes_unq, idx_unq] = unique(surfaceNodes_surf);
% 


%% compute shortest path between two points in the volume
distance4D_surf = nan(numel(yaxis), numel(xaxis), numel(yaxis), numel(xaxis));
path_node4D_surf = cell(numel(yaxis), numel(xaxis), numel(yaxis), numel(xaxis));
tic;
%TODO find surface nodes corresponding to the visual cortex
[ty,tx] = ind2sub([numel(yaxis) numel(xaxis)], withinMask);
for snode=1:numel(surfaceNodes_surf_unq)
     disp(num2str(snode/numel(surfaceNodes_surf_unq)*100));
   [path_nodes, distance_all_unq] = shortestpathtree(Gsurf, ...
       surfaceNodes_surf_unq(snode), ...
       surfaceNodes_surf_unq,'OutputForm','cell');

    distance_all = nan(1,numel(surfaceNodes_surf));
    path_node_all = cell(1,numel(surfaceNodes_surf));
    for tt = 1:numel(surfaceNodes_surf_unq)
        idx = find(surfaceNodes_surf == surfaceNodes_surf_unq(tt));
        distance_all(idx) = distance_all_unq(tt);
        path_node_all(idx) = path_nodes(tt);
    end

    [snodes] = find(surfaceNodes_surf == surfaceNodes_surf_unq(snode));

    for ss = 1:numel(snodes)
        [sy,sx] = ind2sub([numel(yaxis) numel(xaxis)], withinMask(snodes(ss)));
        for tnode=1:numel(surfaceNodes_surf)
            if isnan(distance4D_surf(sy,sx,ty(tnode),tx(tnode)))
                distance4D_surf(sy,sx,ty(tnode),tx(tnode)) = distance_all(tnode);
                path_node4D_surf{sy,sx,ty(tnode),tx(tnode)} = path_node_all{tnode};
            elseif (distance4D_surf(sy,sx,ty(tnode),tx(tnode)) > distance_all(tnode))
                distance4D_surf(sy,sx,ty(tnode),tx(tnode)) = distance_all(tnode);
                path_node4D_surf{sy,sx,ty(tnode),tx(tnode)} = path_node_all{tnode};
            end
        end
    end
end
% distance2D_surf = reshape(distance4D_surf, numel(yaxis)*numel(xaxis), numel(yaxis)*numel(xaxis));
% path_node2D_surf = reshape(path_node4D_surf,numel(yaxis)*numel(xaxis), numel(yaxis)*numel(xaxis));
% t=toc
% 
% 
% save(fullfile(saveDir, subject_id{sid}, ['minimal_path_' type '_hmax' ...
%     num2str(hmax) '_' subject_id{sid}]),'path_node4D_surf','-append','-v7.3');


sxi = 40; syi = 60; %source pixel position on flattend map
txi = [15:10:55]; tyi = 70*ones(1,numel(txi)); %target pixel position on flattend map

vidx_s = xy2node_surf(syi,sxi);%    flat2vert(syi,sxi);
vidx_t = [];
for ii = 1:numel(txi)
    vidx_t(ii) = xy2node_surf(tyi(ii),txi(ii)); %flat2vert(tyi(ii),txi(ii));
end

figure;
subplot(141); %distance on flattened map from (sx,sy)
imagesc(xaxis, yaxis, squeeze(distance4D_surf(syi,sxi,:,:)));hold on;
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
        scatter3(vertices_surf(vidx_s, 1), vertices_surf(vidx_s, 2), vertices_surf(vidx_s, 3), 20, 'r', 'filled');
        scatter3(vertices_surf(vidx_t(ii), 1), vertices_surf(vidx_t(ii), 2), vertices_surf(vidx_t(ii), 3), 20, 'green', 'filled');
    end
    for ii = 1:numel(txi)
        lc = [ii/numel(txi) 0 0];
        plot3(vertices_surf(path_node4D_surf{syi,sxi,tyi(ii),txi(ii)}, 1), ...
            vertices_surf(path_node4D_surf{syi,sxi,tyi(ii),txi(ii)}, 2), ...
            vertices_surf(path_node4D_surf{syi,sxi,tyi(ii),txi(ii)}, 3), '-', 'color',lc,'LineWidth', 2);
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




