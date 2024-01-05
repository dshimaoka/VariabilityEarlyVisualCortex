

% Example usage:
% load('tri_faces_L','tri_faces_L');
% load('mid_pos_L','mid_pos_L');
% faces = tri_faces_L; %[1 2 3; 3 4 5; 5 6 1];  % Replace with your actual face data
% vertices = mid_pos_L; %rand(6, 3);  % Replace with your actual vertex data

% Define the grid
[x, y] = meshgrid(linspace(0, 10, 25), linspace(0, 10, 25));

% Define parameters for the waves
frequency = 1;  % Adjust the frequency of the waves
amplitude = 1;  % Adjust the amplitude of the waves

% Compute the z-values based on a combination of sine and cosine functions
z = amplitude * sin(frequency * x) + amplitude * cos(frequency * y) + 5;

% Create vertices
vertices = [x(:), y(:), z(:)];

% Define faces for a meshgrid
faces = delaunay(x(:), y(:));

%% create a solid volume
fv = surf2solid(faces, vertices); % - not creatin watertight mesh


% Plot the 3D surface
%figure;
trisurf(fv.faces, fv.vertices(:,1), fv.vertices(:,2), fv.vertices(:,3), 'FaceColor', 'interp', 'EdgeColor', 'k');
hold on;
title('3D Surface with Repeated Waves (Faces and Vertices)');
xlabel('X-axis');
ylabel('Y-axis');
zlabel('Z-axis');

%Plot vertices
scatter3(fv.vertices(:, 1), fv.vertices(:, 2), fv.vertices(:, 3), 20, 'k', 'filled','LineWidth',0.1);


%% compute minimal path between two points
start_point = 1;  % starting node index
end_point = 20;    % ending node index

[path_nodes, distance] = compute_minimal_path(fv.faces, fv.vertices, start_point, end_point);
disp('Minimal Path:');
disp(path);
disp('Traveling Distance:');
disp(distance);

scatter3(fv.vertices(start_point, 1), fv.vertices(start_point, 2), fv.vertices(start_point, 3), 20, 'green', 'filled');
scatter3(fv.vertices(end_point, 1), fv.vertices(end_point, 2), fv.vertices(end_point, 3), 20, 'green', 'filled');

plot3(vertices(path_nodes, 1), vertices(path_nodes, 2), vertices(path_nodes, 3), 'r-', 'LineWidth', 2);

%% compute all distance
inode=1;
[~, distance_all, path_nodes_all] = shortestpathtree(G, inode,'OutputForm','cell');

% fv = surf2solid(x,y,z, 'triangulation','x');
% fv = surf2solid(fv.vertices(:,1),fv.vertices(:,2),fv.vertices(:,3));
%rmpath(genpath('/home/daisuke/Documents/git/VariabilityEarlyVisualCortex/matlab/fieldtrip'))
%[node,elem,face] = surf2mesh(vertices, faces, [0 0 0],[1 0 0],1,[]);
% 
% mesh = surfaceMesh(fv.vertices,fv.faces);
% surfaceMeshShow(mesh,Title="Input Mesh")

% export to stl: https://au.mathworks.com/matlabcentral/answers/1716010-input-mesh-does-not-form-a-closed-volume-error-checked-that-mesh-is-watertight
% fv = surf2solid(x,y,z, 'triangulation','x');
% TR = triangulation(fv.faces, fv.vertices);
% %stlwrite('Geom.stl', fv.faces, fv.vertices)
% stlwrite(TR,'Geom.stl'); %/usr/local/MATLAB/R2023a/toolbox/matlab/polyfun/stlwrite.m
% model = createpde;
% gm = importGeometry(model,'Geom.stl')
% pdegplot(gm);

% k = convhull(x,y,z)

% %% make watertight mesh
% % faceNormals = faceNormal(fv.vertices);
% % invertedFaces = find(faceNormals(:, 3) < 0);
% % faces(invertedFaces, :) = faces(invertedFaces, [1 3 2]); % Invert face orientation
% 
% % Step 2: Create a PDE Model
% model = createpde();
% 
% % Add the combined geometry to the model. At this point, the mesh must be
% % watertight
% geometryFromMesh(model, fv.vertices', fv.faces');
% 
% % Generate the tetrahedral mesh
% generateMesh(model);
% 
% 
