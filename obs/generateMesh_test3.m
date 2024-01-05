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

% Create a base on z=0 by connecting the vertices in a rectangular pattern
base_vertices = [xGrid(:), yGrid(:), zGrid(:)];
base_faces = delaunay(xGrid(:), yGrid(:));

% Combine all vertices and faces
all_vertices = [vertices_surface; vertices_z0; base_vertices];
all_faces = [faces_surface; 
             size(vertices_surface, 1) + delaunay(xGrid(:), yGrid(:)); 
             size(vertices_surface, 1) + size(vertices_z0, 1) + base_faces];

% Step 2: Create a PDE Model
model = createpde();

% Add the combined geometry to the model
geometryFromMesh(model, all_vertices', all_faces');

% Generate the mesh
generateMesh(model);
