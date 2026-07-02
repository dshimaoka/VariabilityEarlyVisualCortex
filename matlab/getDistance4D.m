function [distance4D, path_node4D, surfaceNodes, xy2node] ...
    = getDistance4D(G, grid_x, grid_y, array_3d)

[x,y,z,xaxis, yaxis, withinMask] = getXYZ(grid_x, grid_y, array_3d);
P = G.Nodes;
if istable(P)
    P = table2array(P);
end

surfaceNodes = zeros(numel(x),1);
for idx = 1:numel(x)
    [~,surfaceNodes(idx)] = min(abs(P(:,1) - x(idx)).^2 ...
        + abs(P(:,2) - y(idx)).^2 ...
        + abs(P(:,3) - z(idx)).^2);
end

[surfaceNodes_surf_unq, idx_unq] = unique(surfaceNodes);

xy2node = nan(numel(yaxis),numel(xaxis));
xy2node(withinMask) = surfaceNodes;



%% compute shortest path between two points in the volume
distance4D = nan(numel(yaxis), numel(xaxis), numel(yaxis), numel(xaxis));
path_node4D = cell(numel(yaxis), numel(xaxis), numel(yaxis), numel(xaxis));
tic;
%TODO find surface nodes corresponding to the visual cortex
[ty,tx] = ind2sub([numel(yaxis) numel(xaxis)], withinMask);
f = waitbar(0,'computing shortest path distance');
for snode=1:numel(surfaceNodes_surf_unq)
    % disp(num2str(snode/numel(surfaceNodes_surf_unq)*100));
    waitbar(snode/numel(surfaceNodes_surf_unq),f);
    [path_nodes, distance_all_unq] = shortestpathtree(G, ...
        surfaceNodes_surf_unq(snode), ...
        surfaceNodes_surf_unq,'OutputForm','cell');

    distance_all = nan(1,numel(surfaceNodes));
    path_node_all = cell(1,numel(surfaceNodes));
    for tt = 1:numel(surfaceNodes_surf_unq)
        idx = find(surfaceNodes == surfaceNodes_surf_unq(tt));
        distance_all(idx) = single(distance_all_unq(tt));
        path_node_all(idx) = cellfun(@single, path_nodes(tt), 'UniformOutput', false);
    end

    [snodes] = find(surfaceNodes == surfaceNodes_surf_unq(snode));

    for ss = 1:numel(snodes)
        [sy,sx] = ind2sub([numel(yaxis) numel(xaxis)], withinMask(snodes(ss)));
        for tnode=1:numel(surfaceNodes)
            if isnan(distance4D(sy,sx,ty(tnode),tx(tnode)))
                distance4D(sy,sx,ty(tnode),tx(tnode)) = distance_all(tnode);
                path_node4D{sy,sx,ty(tnode),tx(tnode)} = path_node_all{tnode};
            elseif (distance4D(sy,sx,ty(tnode),tx(tnode)) > distance_all(tnode))
                distance4D(sy,sx,ty(tnode),tx(tnode)) = distance_all(tnode);
                path_node4D{sy,sx,ty(tnode),tx(tnode)} = path_node_all{tnode};
            end
        end
    end
end
close(f);
% distance2D_surf = reshape(distance4D_surf, numel(yaxis)*numel(xaxis), numel(yaxis)*numel(xaxis));
% path_node2D_surf = reshape(path_node4D_surf,numel(yaxis)*numel(xaxis), numel(yaxis)*numel(xaxis));
% t=toc
%
%
% save(fullfile(saveDir, subject_id{sid}, ['minimal_path_' type '_hmax' ...
%     num2str(hmax) '_' subject_id{sid}]),'path_node4D_surf','-append','-v7.3');
