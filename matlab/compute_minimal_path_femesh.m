% %% import mask from export_geometry_test.py
load('array_3d.mat','array_3d','grid_x','grid_y');%,'mask');
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
figure;plot3(x,y,z,'.');xlabel('x');ylabel('y');zlabel('z');


%% 1. import entire brain
tic
hmax = 2; %1: too slow, 3: too coarse
mid_gifti_L=gifti('S1200_7T_Retinotopy181.L.midthickness_MSMAll.32k_fs_LR.surf.gii');
stlwrite('Geom.stl', mid_gifti_L.faces, mid_gifti_L.vertices)
model = createpde(1);
importGeometry(model,"Geom.stl"); %"BracketTwoHoles.stl");%
generateMesh(model,"Hmax",hmax); %determines coarseness of the mesh
pdeplot3D(model)
t1=toc

%% 2. extract faces and vertices from FEMesh
% The vertices array will contain the coordinates of all vertices in the mesh.
% The faces array will have the indices of vertices that define each face of every element in the mesh.
% The unique_faces array will give you a list of unique faces in the mesh.

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
t2=toc

%% 3. convert FEMesh to graph
tic
% Assuming your FEMesh object has nodes and elements properties
nodes = model.Mesh.Nodes';
elements = model.Mesh.Elements';

% Create an adjacency matrix based on element connectivity
num_nodes = size(nodes, 1);

% strategy1: create adjacency matrix - SLOW
% Loop through all elements to populate adjacency matrix 
%SLOW
%adj_matrix = sparse(num_nodes, num_nodes);
%for i = 1:size(elements, 1)
    %elem = elements(i, :);
    % % Define edges based on element connectivity
    %edges = nchoosek(elem, 2);
    % % Update adjacency matrix for each edge
    % for j = 1:size(edges, 1)
    %     node1 = edges(j, 1);
    %     node2 = edges(j, 2);
    % 
    %     % Extract coordinates of the two nodes
    %     coord1 = nodes(node1, :);
    %     coord2 = nodes(node2, :);
    % 
    %     % Compute Euclidean distance
    %     distance = norm(coord2 - coord1);
    % 
    %     adj_matrix(node1, node2) = distance;
    %     %adj_matrix(node2, node1) = distance;  % Assuming undirected graph
    % end
%end
% % Create a graph object from the adjacency matrix
% adj_matrix = spalloc(num_nodes, num_nodes, 2*sum(cellfun(@numel, index)));
% adj_matrix(index_all) = distance_all; %SLOW
% adj_matrix = adj_matrix + adj_matrix';
% G = graph(adj_matrix);

%% strategy 2:   Create a graph object from edges and weights
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

t3 = toc


%% obtain nodes on the surface 
surfaceNodes = zeros(numel(x),1);
for idx = 1:numel(x)
    [~,surfaceNodes(idx)] = min(abs(nodes(:,1) - x(idx)).^2 + abs(nodes(:,2) - y(idx)).^2 + abs(nodes(:,3) - z(idx)).^2);
end

[surfaceNodes_unq, idx_unq] = unique(surfaceNodes); 

xy2node = nan(numel(yaxis),numel(xaxis));
xy2node(withinMask) = surfaceNodes;
% xy2node(withinMask(idx_unq)) = surfaceNodes(idx_unq);
%idx_dub = setdiff(1:numel(surfaceNodes), idx_unq)';
%xy2node(withinMask(idx_dub)) = nan;

%% Use Dijkstra's algorithm to find the shortest path
start_point = surfaceNodes(1);
end_point = surfaceNodes(600); 
[path_nodes, distance] = compute_minimal_path_fromGraph(G, vertices, start_point, end_point);

disp('Minimal Path:');
disp(path_nodes);
disp('Traveling Distance:');
disp(distance);

%plot 3d vertices and shortest path
figure;
trisurf(faces, vertices(:,1), vertices(:,2), vertices(:,3), 'FaceColor', 'interp', 'EdgeColor', 'k','facealpha',.9, 'edgealpha',.1); hold on;
scatter3(vertices(start_point, 1), vertices(start_point, 2), vertices(start_point, 3), 20, 'green', 'filled');
scatter3(vertices(end_point, 1), vertices(end_point, 2), vertices(end_point, 3), 20, 'green', 'filled');
plot3(vertices(path_nodes, 1), vertices(path_nodes, 2), vertices(path_nodes, 3), 'r-', 'LineWidth', 2);
xlabel('x'); ylabel('y');
axis equal tight;
hold off;

%% compute all distance
distance4D = nan(numel(yaxis), numel(xaxis), numel(yaxis), numel(xaxis));
distance2D = nan(numel(surfaceNodes));
for snode=1:numel(surfaceNodes_unq)
    disp(snode)
    [~, distance_all_unq] = shortestpathtree(G, surfaceNodes_unq(snode), surfaceNodes_unq);%'OutputForm','cell');
    
    distance_all = nan(1,numel(surfaceNodes));
    for tt = 1:numel(surfaceNodes_unq)
        idx = find(surfaceNodes == surfaceNodes_unq(tt));
        distance_all(idx) = distance_all_unq(tt);
    end

    [snodes] = find(surfaceNodes == surfaceNodes_unq(snode));

    [ty,tx] = ind2sub([numel(yaxis) numel(xaxis)], withinMask);
    for ss = 1:numel(snodes)
        [sy,sx] = ind2sub([numel(yaxis) numel(xaxis)], withinMask(snodes(ss)));
        for tnode=1:numel(surfaceNodes)
            distance4D(sy,sx,ty(tnode),tx(tnode)) = distance_all(tnode);
        end
        distance2D(snodes(ss),:) = distance_all;
    end
    % imagesc(squeeze(distance4D(sy,sx,:,:))); hold on; plot(sx,sy,'ro')
end

save(['minimal_path_hmax' num2str(hmax)],'distance4D','distance2D','xy2node','surfaceNodes');%,'G','-v7.3');

%% show surface and shortest path distance
xidx = 54; yidx = 63; %61

figure('position',[0 0 1900 1000]);

subplot(122);
imagesc(xaxis, yaxis, squeeze(distance4D(yidx,xidx,:,:)));hold on;
scatter(xaxis(xidx), yaxis(yidx), 50, 'filled', 'MarkerFaceColor', 'r');
xlabel('x'); ylabel('y');
title('shortest path distance');        
colorbar;
axis equal tight xy;
clim([0 20])

subplot(1,2,1);
trisurf(faces, vertices(:,1), vertices(:,2), vertices(:,3), 'FaceColor', 'c', 'EdgeColor', 'k'); hold on;
%scatter3(vertices(find(mask), 1), vertices(find(mask), 2), vertices(find(mask), 3), 10, 'b', 'filled');
scatter3(x,y,z,2,'w.');
scatter3(vertices(xy2node(yidx,xidx), 1), vertices(xy2node(yidx,xidx), 2), vertices(xy2node(yidx,xidx), 3), 50, 'r', 'filled');
xlabel('x'); ylabel('y');
%view(20, -30);
title('surface');
axis equal tight off;

% Set the range of rotation angles
angles = -90:2:90;  % Change the increment (5 degrees) as needed

% Initialize GIF file
filename = 'rotating_3d_figure.gif';

% Loop through each angle and capture frame for the GIF
for i = 1:length(angles)
    view([angles(i), 0]);  
    
    % Capture the current figure as an image
    frame = getframe(gcf);
    im = frame2im(frame);
    [imind, cm] = rgb2ind(im, 256);
    
    % Write to the GIF File
    if i == 1
        imwrite(imind, cm, filename, 'gif', 'Loopcount', inf, 'DelayTime', 0.1);
    else
        imwrite(imind, cm, filename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
    end
end


screen2png(['minimal_path_hmax' num2str(hmax)])