function [linearIndex, sourcenode, tgtnode, distance] = computeAdjacency(elem, nodes, size)

    % Define edges based on element connectivity
    edges = nchoosek(elem, 2);
    sourcenode = edges(:,1);
    tgtnode = edges(:,2);


    linearIndex = sub2ind(size,sourcenode,tgtnode);
    difs = nodes(sourcenode,:) - nodes(tgtnode,:);
    distance = sqrt(difs(:,1).^2 + difs(:,2).^2 + difs(:,3).^2);
    %adj_matrix(index) = distance;


       %elem = elements(i, :);
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

