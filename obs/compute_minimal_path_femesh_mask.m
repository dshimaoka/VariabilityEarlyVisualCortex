%% construct 3D surface
% % Define the grid
% xaxis = linspace(0, 10, 25);
% yaxis = linspace(0, 8, 20);
% [x, y] = meshgrid(xaxis, yaxis);
% 
% % Define parameters for the waves
% frequency = 1;  % Adjust the frequency of the waves
% amplitude = 2;  % Adjust the amplitude of the waves
% 
% % Compute the z-values based on a combination of sine and cosine functions
% z = amplitude * sin(frequency * x) + amplitude * cos(frequency * y) + 5;
% fv = surf2solid(x,y,z, 'triangulation','x');
% 
% %% export to stl
% % https://au.mathworks.com/matlabcentral/answers/1716010-input-mesh-does-not-form-a-closed-volume-error-checked-that-mesh-is-watertight
% stlwrite('Geom.stl', fv.faces, fv.vertices)
% 
% 
% %% define FEMesh
% %https://au.mathworks.com/help/pde/ug/generate-mesh.html
% %https://au.mathworks.com/help/pde/ug/pde.pdemodel.generatemesh.html
% 
% model = createpde(1);
% importGeometry(model,"Geom.stl"); %"BracketTwoHoles.stl");%


% %% import from snipped brain
load('array_3d.mat','array_3d','grid_x','grid_y','mask');
xaxis = squeeze(grid_x(:,1))';
yaxis = squeeze(grid_y(1,:));
x = reshape(array_3d(:,:,1),numel(xaxis)*numel(yaxis),1);
y = reshape(array_3d(:,:,2),numel(xaxis)*numel(yaxis),1);
z = reshape(array_3d(:,:,3),numel(xaxis)*numel(yaxis),1);
%omit pixels outside the mask
mask_in2d = find(~isnan(x));
x = x(mask_in2d); 
y = y(mask_in2d); 
z = z(mask_in2d); 
figure;plot3(x,y,z,'.');xlabel('x');ylabel('y');zlabel('z');


% % Create a 3D Delaunay triangulation ... NG as it will fill convex
% DT = delaunayTriangulation(x(:), y(:), z(:));
% 
% model = createpde(1);
% geometryFromMesh(model,DT.Points',DT.ConnectivityList');
% generateMesh(model); %determines coarseness of the mesh
% pdeplot3D(model)

%% import entire brain
g=gifti('S1200_7T_Retinotopy181.L.midthickness_MSMAll.32k_fs_LR.surf.gii');


%% only retain mask region
[faces_mask, vertices_mask] = getMaskedSurface(g.faces, g.vertices, mask);
patch('Faces', faces_mask, 'Vertices', vertices_mask, 'FaceColor', 'blue', 'EdgeColor', 'none', 'FaceAlpha', 0.7);

%%% create a new closed volume w boundary.m
% vertices_mask = double(vertices_mask);
% vertices_mask_c = double([vertices_mask; median(vertices(boundary_edges(:),:))]);
% bb=boundary(vertices_mask_c, 0);
% trisurf(single(bb),vertices_mask_c(:,1),vertices_mask_c(:,2),vertices_mask_c(:,3));%,'red','FaceAlpha',0.3)

%% create surface around the rim of the mask
masked_vertices_idx = find(mask);
unmasked_vertices_idx = setdiff(1:size(g.vertices, 1),  masked_vertices_idx);

% 3. Identify Boundary Edges
boundary_edges = [];
faces = g.faces;
for i = 1:size(faces, 1)
    face = faces(i, :);
    if any(ismember(face, masked_vertices_idx)) && any(ismember(face, unmasked_vertices_idx))
        for j = 1:3
            vertex1 = face(j);
            vertex2 = face(mod(j, 3) + 1);
            if (ismember(vertex1, masked_vertices_idx) && ismember(vertex2, unmasked_vertices_idx)) || ...
               (ismember(vertex2, masked_vertices_idx) && ismember(vertex1, unmasked_vertices_idx))
                edge = sort([vertex1, vertex2]);
                boundary_edges = [boundary_edges; edge];
            end
        end
    end
end

% 4. Extract the Outer Rim Vertices
% Remove duplicate edges
boundary_edges = unique(sort(boundary_edges, 2), 'rows', 'stable');
unique_edges = unique(boundary_edges(:));
omitSub = [];
for ii = 1:numel(unique_edges)
    idx = find(boundary_edges == unique_edges(ii));
    if numel(idx)==1
        [rr,cc]=ind2sub(size(boundary_edges),idx);
        omitSub = [omitSub rr];
    end
end
boundary_edges(omitSub,:)=[];

% % create new volume w addFace
% g2=fegeometry(vertices_mask, faces_mask);
% h = addFace(g2, unique(boundary_edges(:)));

%model_mask = createpde('structural');
%model_mask = geometryFromEdges(model_mask, vertices_mask,faces_mask);

%% assign new index
[be_s] = sort(unique(boundary_edges(:)));
boundary_edges_new = nan(size(boundary_edges));
for ii = 1:size(boundary_edges,1)
    for jj = 1:size(boundary_edges,2)
        boundary_edges_new(ii,jj) = find(be_s == boundary_edges(ii,jj));
    end
end

% Step 4: Create new faces to close the boundary and form a closed volume
% For simplicity, let's try to connect the boundary edges to form a closed volume.
% This may require more sophisticated algorithms based on specific requirements.

% Here, we'll attempt a simple method:
% Connect the boundary edges to form triangles (assuming a simple approach for illustration)
% NOT WORKING!!

new_faces = [];
for i = 1:size(boundary_edges, 1)
    for j = (i+1):size(boundary_edges, 1)
        edge1 = boundary_edges_new(i, :);
        edge2 = boundary_edges_new(j, :);
        common_vertex = intersect(edge1, edge2);
        if ~isempty(common_vertex)
            vertex_to_connect = setdiff([edge1, edge2], common_vertex);
            new_faces = [new_faces; common_vertex, edge1(find(edge1 ~= common_vertex)), edge2(find(edge2 ~= common_vertex))];
        end
    end
end


for ib = 1:size(boundary_edges,1)
    plot3(vertices(boundary_edges(ib,:),1),vertices(boundary_edges(ib,:),2),vertices(boundary_edges(ib,:),3))
    hold on
end

patch('Faces', new_faces, 'Vertices', vertices(boundary_edges, :), ... 
      'FaceColor', 'blue', 'EdgeColor', 'none', 'FaceAlpha', 0.7);

% outer_rim_vertices = unique(boundary_edges(:));
% outer_rim_vertices_idx = (1:numel(outer_rim_vertices))';
%
% num_vertices = length(outer_rim_vertices);
% faces_outer_rim = [outer_rim_vertices_idx(1:end-1), outer_rim_vertices_idx(2:end), outer_rim_vertices_idx(2:end) + num_vertices;
%          outer_rim_vertices_idx(1), outer_rim_vertices_idx(end), outer_rim_vertices_idx(end) + num_vertices];
% 
% % Create a figure and plot the unclosed surface using patch
% figure;
% patch('Faces', faces_outer_rim, 'Vertices', vertices(outer_rim_vertices, :), ... 
%       'FaceColor', 'blue', 'EdgeColor', 'none', 'FaceAlpha', 0.7);

%NG creating closed volume
% xx=double(vertices(outer_rim_vertices, 1)); 
% yy=double(vertices(outer_rim_vertices, 2)); 
% zz=double(vertices(outer_rim_vertices, 3));
% DT = delaunayTriangulation(xx,yy,zz);

xx=double(vertices(unique(boundary_edges(:)), 1)); 
yy=double(vertices(unique(boundary_edges(:)), 2)); 
zz=double(vertices(unique(boundary_edges(:)), 3));
DT = delaunayTriangulation(xx,yy,zz);
tetramesh(DT,'FaceAlpha',0.3);



%% export to .stl
% stlwrite('Geom.stl', g.faces, g.vertices)
%stlwrite('Geom.stl', updated_faces, updated_vertices) %not a closed volume
model = createpde(1);
importGeometry(model,"BracketTwoHoles.stl");%
generateMesh(model,"Hmax",3); %determines coarseness of the mesh
pdeplot3D(model)


%% extract faces and vertices from FEMesh
% The vertices array will contain the coordinates of all vertices in the mesh.
% The faces array will have the indices of vertices that define each face of every element in the mesh.
% The unique_faces array will give you a list of unique faces in the mesh.

nodes = model.Mesh.Nodes';
elements = model.Mesh.Elements';

% Extract vertices (nodes)
vertices = nodes;

% Initialize an empty array to store faces
faces = [];

% Loop through all elements to extract faces
for i = 1:size(elements, 1)
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


%% convert FEMesh to graph
% Assuming your FEMesh object has nodes and elements properties
nodes = model.Mesh.Nodes';
elements = model.Mesh.Elements';

% Create an adjacency matrix based on element connectivity
num_nodes = size(nodes, 1);
%adj_matrix = zeros(num_nodes, num_nodes);
adj_matrix = sparse(num_nodes, num_nodes);

% Loop through all elements to populate adjacency matrix 
%SLOW
for i = 1:size(elements, 1)
    elem = elements(i, :);
    
    % Define edges based on element connectivity
    edges = nchoosek(elem, 2);
    
    % Update adjacency matrix for each edge
    for j = 1:size(edges, 1)
        node1 = edges(j, 1);
        node2 = edges(j, 2);
        
        % Extract coordinates of the two nodes
        coord1 = nodes(node1, :);
        coord2 = nodes(node2, :);
    
        % Compute Euclidean distance
        distance = norm(coord2 - coord1);

        adj_matrix(node1, node2) = distance;
        adj_matrix(node2, node1) = distance;  % Assuming undirected graph
    end
end

% Create a graph object from the adjacency matrix
G = graph(adj_matrix);

% Plot the graph (optional)
% figure;
% plot(G, 'NodeLabel', {});
% title('Graph from FEMesh');

%% obtain nodes on the surface (NOT FUNCTIONAL)
surfaceNodes = zeros(numel(x),1);
for idx = 1:numel(x)
    [~,surfaceNodes(idx)] = min(abs(nodes(:,1) - x(idx)).^2 + abs(nodes(:,2) - y(idx)).^2 + abs(nodes(:,3) - z(idx)).^2);
end

xy2node = nan(numel(yaxis),numel(xaxis));
xy2node(mask) = surfaceNodes;


%% Use Dijkstra's algorithm to find the shortest path
start_point = surfaceNodes(1);
end_point = surfaceNodes(600); 
[path_nodes, distance] = compute_minimal_path_fromGraph(G, vertices, start_point, end_point);

disp('Minimal Path:');
disp(path_nodes);
disp('Traveling Distance:');
disp(distance);

%plot 3d vertices and shortest path
%figure;
trisurf(faces, vertices(:,1), vertices(:,2), vertices(:,3), 'FaceColor', 'interp', 'EdgeColor', 'k','facealpha',.9, 'edgealpha',.1); hold on;
scatter3(vertices(start_point, 1), vertices(start_point, 2), vertices(start_point, 3), 20, 'green', 'filled');
scatter3(vertices(end_point, 1), vertices(end_point, 2), vertices(end_point, 3), 20, 'green', 'filled');
plot3(vertices(path_nodes, 1), vertices(path_nodes, 2), vertices(path_nodes, 3), 'r-', 'LineWidth', 2);
xlabel('x'); ylabel('y');
axis equal tight;
hold off;

%% compute all distance
distance4D = nan(numel(yaxis), numel(xaxis), numel(yaxis), numel(xaxis));
for snode=1:numel(surfaceNodes)
    [~, distance_all] = shortestpathtree(G, surfaceNodes(snode), surfaceNodes);%'OutputForm','cell');
    
    [sy,sx] = ind2sub([numel(yaxis) numel(xaxis)], mask(snode));
    [ty,tx] = ind2sub([numel(yaxis) numel(xaxis)], mask);

    for tnode=1:numel(surfaceNodes)
        distance4D(sy,sx,ty,tx) = distance_all(tnode);
    end
end

%distance4D = reshape(distance_all, numel(yaxis), numel(xaxis), numel(yaxis), numel(xaxis));


%% show surface and shortest path distance
xidx = 1; yidx = 1;

figure;
subplot(121);
trisurf(faces, vertices(:,1), vertices(:,2), vertices(:,3), 'FaceColor', 'interp', 'EdgeColor', 'none'); hold on;
scatter3(vertices(xy2node(yidx,xidx), 1), vertices(xy2node(yidx,xidx), 2), vertices(xy2node(yidx,xidx), 3), 50, 'r', 'filled');
xlabel('x'); ylabel('y');
view(0, 90);
title('surface');
colorbar;
axis equal tight;

subplot(122);
imagesc(xaxis, yaxis, squeeze(distance4D(yidx,xidx,:,:)));hold on;
scatter(xaxis(xidx), yaxis(yidx), 50, 'filled', 'MarkerFaceColor', 'r');
xlabel('x'); ylabel('y');
title('shortest path distance');        
colorbar;
axis equal tight xy;
