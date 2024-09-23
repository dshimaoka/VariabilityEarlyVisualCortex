subject_id = {'157336','585256','114823','581450','725751'};

loadDir = '/home/daisuke/Documents/git/VariabilityEarlyVisualCortex/data/';
saveDir = '/home/daisuke/Documents/git/VariabilityEarlyVisualCortex/results/';

for ii = 2%:numel(subject_id)
    % %% import mask from export_geometry_test.py
    %load('array_3d.mat','array_3d','grid_x','grid_y');%,'mask');
    load(fullfile(loadDir, ['geometry_retinotopy_'  subject_id{ii}   '.mat']),'array_3d','grid_x','grid_y');
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
    type = 'midthickness';%'white' %cannot generateMesh with 'pial'
    hmax = 2; %1: fine but too slow, 3: too coarse
    load(fullfile(loadDir, ['tri_faces_L_' subject_id{ii}]), 'tri_faces_L')
    load(fullfile(loadDir, ['mid_pos_L_' subject_id{ii}]), 'mid_pos_L');
        stlwrite(fullfile(saveDir, ['Geom_'  subject_id{ii} '.stl']), tri_faces_L, mid_pos_L)
        model = createpde(1);
    %model = createpde('structural','static-solid');
    importGeometry(model, fullfile(saveDir, ['Geom_'  subject_id{ii} '.stl'])); %"BracketTwoHoles.stl");%
    %pdegplot(model)
  
    % generateMesh(model,"Hmax",hmax);%,"geometricOrder","linear","Hmin",0.2*mm); %determines coarseness of the mesh
generateMesh(model,"geometricOrder","linear","Hmin",.01);
    t1=toc
     pdeplot3D(model)

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


    %% obtain nodes on the surface ... its order is according to order of x ... completely random
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
    screen2png(fullfile(loadDir, ['minimal_path_' type '_hmax' num2str(hmax) '_' subject_id{ii}]));

    %% compute all distance
    distance4D = nan(numel(yaxis), numel(xaxis), numel(yaxis), numel(xaxis));
    for snode=1:numel(surfaceNodes_unq)
        disp(snode)
        [~, distance_all_unq] = shortestpathtree(G, surfaceNodes_unq(snode), surfaceNodes_unq);

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
            %distance2D(snodes(ss),:) = distance_all;
        end
        % imagesc(squeeze(distance4D(sy,sx,:,:))); hold on; plot(sx,sy,'ro')
    end
    distance2D = reshape(distance4D, numel(yaxis)*numel(xaxis), numel(yaxis)*numel(xaxis));

    %% Euclidean distance
    distance4D_euc = nan(numel(yaxis), numel(xaxis), numel(yaxis), numel(xaxis));
    X=repmat(x, [1,numel(x)]);
    Y=repmat(y, [1,numel(y)]);
    Z=repmat(z, [1,numel(z)]);
    distance_euc_c = sqrt((X-X').^2+(Y-Y').^2+(Z-Z').^2);
    [sy,sx] = ind2sub([numel(yaxis) numel(xaxis)], withinMask);
    [ty,tx] = ind2sub([numel(yaxis) numel(xaxis)], withinMask);
    for ii = 1:numel(withinMask)
        for jj = 1:numel(withinMask)
            distance4D_euc(sy(ii), sx(ii), ty(jj), tx(jj)) = distance_euc_c(ii,jj);
        end
    end
    distance2D_euc = reshape(distance4D_euc, numel(yaxis)*numel(xaxis), numel(yaxis)*numel(xaxis));

    save(fullfile(loadDir, ['minimal_path_' type '_hmax' num2str(hmax) '_' subject_id{ii}]), ...
        'distance4D','distance2D','xy2node','surfaceNodes',...
        'distance4D_euc','distance2D_euc');%,'G','-v7.3');
end


