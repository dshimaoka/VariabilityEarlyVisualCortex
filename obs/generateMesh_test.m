% Step 1: Define the Geometry
% Create a 3D surface (e.g., a sphere)
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

% Create a convex hull for the z=0 vertices to get a triangular representation
k = convhull(vertices_z0(:,1), vertices_z0(:,2));
faces_z0 = k(:, [1, 2, 3]); % Triangular faces from convex hull

% Combine all vertices and faces
all_vertices = [vertices_surface; vertices_z0];
all_faces = [faces_surface; size(vertices_surface, 1) + faces_z0];

% Step 2: Create a PDE Model
model = createpde();

% Add the combined geometry to the model
geometryFromMesh(model, all_vertices', all_faces');

% Generate the mesh
generateMesh(model);
