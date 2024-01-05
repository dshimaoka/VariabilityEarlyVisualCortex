function [path, distance] = compute_minimal_path(faces, vertices, start_point, end_point)
%[path, distance] = compute_minimal_path(faces, vertices, start_point, end_point)
% computes minimal path and traveling distance
% faces: N x 3
% vertices: M x 3

% Function to compute Euclidean distance between two points
euclidean_distance = @(point1, point2) norm(point2 - point1);

% Create a graph representing the surface
    G = create_surface_graph_gpu(faces, vertices);

    % Use Dijkstra's algorithm to find the shortest path
    [path_nodes, ~] = shortestpath(G, start_point, end_point);

    % Compute the traveling distance of the path
    distance = 0;
    for i = 1:length(path_nodes)-1
        distance = distance + euclidean_distance(vertices(path_nodes(i), :), vertices(path_nodes(i+1), :));
    end

    % Output the path as indices and the total traveling distance
    path = path_nodes;
end

%Function to create a graph representing the surface
function G = create_surface_graph(faces, vertices)
% Function to compute Euclidean distance between two points
% SLOW
euclidean_distance = @(point1, point2) norm(point2 - point1);

G = graph();

    % Add nodes for each vertex
    for i = 1:size(vertices, 1)
        G = addnode(G, i);
    end

    % Add edges based on the faces
    for i = 1:size(faces, 1)
        face = faces(i, :);
        for j = 1:length(face)
            for k = j+1:length(face)
                G = addedge(G, face(j), face(k), euclidean_distance(vertices(face(j), :), vertices(face(k), :)));
            end
        end
    end
end

% Function to create a graph representing the surface using GPU arrays
function G = create_surface_graph_gpu(faces, vertices)
    G = graph();

    % Add nodes for each vertex
    for i = 1:size(vertices, 1)
        G = addnode(G, i); %slow
    end

    % Transfer data to GPU
    vertices_gpu = gpuArray(vertices);

    % Add edges based on the faces using GPU arrays
    for i = 1:size(faces, 1)
        face = faces(i, :);
        local_edges = [];
        for j = 1:length(face)
            for k = j+1:length(face)
                % Compute distance using GPU arrays
                local_edges = [local_edges; face(j), face(k), gather(norm(vertices_gpu(face(j), :) - vertices_gpu(face(k), :)))];
            end
        end
        G = addedge(G, local_edges(:, 1), local_edges(:, 2), local_edges(:, 3));
    end
end

% function G = create_surface_graph_parallel(faces, vertices)
% SHOULD NOT WORK
%     G = graph();
% 
%     % Add nodes for each vertex
%     for i = 1:size(vertices, 1)
%         G = addnode(G, i);
%     end
% 
%     % Add edges based on the faces using parallel computing
%     parfor i = 1:size(faces, 1)
%         face = faces(i, :);
%         local_edges = [];
%         for j = 1:length(face)
%             for k = j+1:length(face)
%                 local_edges = [local_edges; face(j), face(k), norm(vertices(face(j), :) - vertices(face(k), :))];
%             end
%         end
%         G = addedge(G, local_edges(:, 1), local_edges(:, 2), local_edges(:, 3));
%     end
% end