function [path, distance] = compute_minimal_path_fromGraph(G, vertices, start_point, end_point)
%[path, distance] = compute_minimal_path(faces, vertices, start_point, end_point)
% computes minimal path and traveling distance
% faces: N x 3
% vertices: M x 3

% Function to compute Euclidean distance between two points
euclidean_distance = @(point1, point2) norm(point2 - point1);


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