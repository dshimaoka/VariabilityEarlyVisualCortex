% Create a 3D shape (e.g., a sphere)
[X, Y, Z] = sphere(20); % Generate sphere points
Z = Z * 2; % Scale the sphere

% Define the vertices of the surface
vertices_surface = [X(:), Y(:), Z(:)];

% Define the faces of the surface (triangular faces)
faces_surface = delaunay(X(:), Y(:));

% For the z=0 face, create a grid of points and extract the boundary
[xGrid, yGrid] = meshgrid(linspace(min(X(:)), max(X(:)), 50), ...
                           linspace(min(Y(:)), max(Y(:)), 50));
zGrid = zeros(size(xGrid));

% Vertices for z=0 face
vertices_z0 = [xGrid(:), yGrid(:), zGrid(:)];

% Create a boundary loop on the z=0 plane using convhull
k = convhull(vertices_z0(:,1), vertices_z0(:,2));
boundary_loop = vertices_z0(k, :);

% Combine all vertices and faces
all_vertices = [vertices_surface; vertices_z0; boundary_loop];
all_faces = [faces_surface; 
             size(vertices_surface, 1) + delaunay(xGrid(:), yGrid(:)); 
             size(vertices_surface, 1) + size(vertices_z0, 1) + convhull(boundary_loop(:, 1), boundary_loop(:, 2))];

% Visualization (Optional)
figure;
trisurf(all_faces, all_vertices(:,1), all_vertices(:,2), all_vertices(:,3), 'FaceColor', 'cyan', 'EdgeColor', 'none');
axis equal;
xlabel('X');
ylabel('Y');
zlabel('Z');
title('Watertight Mesh Visualization');
