% subject_id = {'avg','114823','157336','585256','581450','725751'};
subject_id = getSubjectId;%setxor(getSubjectId, {'114823','157336','585256','581450','725751'});
%114823: meshing failed for hmax of 2 and hmin of 1

%loadDir = '/home/daisuke/Documents/git/VariabilityEarlyVisualCortex/data/';
saveDir = '/mnt/dshi0006_market/VariabilityEarlyVisualCortex/';
saveDir_tmp = '~/tmp/';

type = 'midthickness';%'white' %cannot generateMesh with 'pial'
hmax = 2; %1: fine but too slow, 3: too coarse

% source and target voxels for sanity check
sxi = 40; syi = 60; %source pixel position on flattend map
txi = [15:10:55]; tyi = 70*ones(1,numel(txi)); %target pixel position on flattend map

for sid = 1:numel(subject_id)

    try
    % from export_geometry_individual.py
    load(fullfile(saveDir, subject_id{sid}, ['geometry_retinotopy_'  subject_id{sid}   '.mat']),...
        'array_3d','grid_x','grid_y', 'grid_curv');%,'mask');
    [x,y,z,xaxis, yaxis, withinMask] = getXYZ(grid_x, grid_y, array_3d);

    %% 0. import entire brain
    tic
    model = createpde(1);
    importGeometry(model, fullfile(saveDir, subject_id{sid}, ['Geom_'  ...
        subject_id{sid} '_hclaplacian.stl'])); %"BracketTwoHoles.stl");%
    %pdegplot(model)

    generateMesh(model,"Hmax",hmax);%,"geometricOrder","linear","Hmin",0.2*mm); %determines coarseness of the mesh
    t0=toc %20s


    %% %%%%%%%%%%% PATH MINIMIZATION ALONG CORTICAL SURFACE %%%%%%%%%%%%%%%%
    %% 1' compute surface mesh
    TR = triangulation(model.Mesh.Elements(1:4,:)', model.Mesh.Nodes');
    [Face_s, Vertex_s] = freeBoundary(TR);
    Face_s = single(Face_s);
    Vertex_s = single(Vertex_s);

    %% 2' convert surface mesh to graph - instanteneous
    tic
    edges_surf = [
        Face_s(:,[1 2]);
        Face_s(:,[2 3]);
        Face_s(:,[3 1]);
        ];

    % Remove duplicate edges
    edges_surf = sort(edges_surf,2);
    edges_surf = unique(edges_surf,'rows');

    % create a weighted graph
    v1 = Vertex_s(edges_surf(:,1),:);
    v2 = Vertex_s(edges_surf(:,2),:);

    weights = sqrt(sum((v1-v2).^2,2));

    G_s = graph(edges_surf(:,1), edges_surf(:,2), weights);
    G_s.Nodes.X = Vertex_s(:,1);
    G_s.Nodes.Y = Vertex_s(:,2);
    G_s.Nodes.Z = Vertex_s(:,3);

    %% 3. compute shortest path distance (surface)
    tic
    [distance4D_s, path_node4D_s, surfaceNodes_s, xy2node_s] ...
        = getDistance4D(G_s, grid_x, grid_y, array_3d);
    distance2D_s = reshape(distance4D_s, numel(yaxis)*numel(xaxis), numel(yaxis)*numel(xaxis));
    t3s = toc %20s

    %% 4. sanity check
    f_surf = examinePathNode(Vertex_s, distance4D_s, path_node4D_s, xaxis, yaxis, ...
        xy2node_s, sxi, syi, txi, tyi, Face_s, Vertex_s, 0.8);
    screen2png(fullfile(saveDir, subject_id{sid}, ['surface_minimal_path_hmax' num2str(hmax) '_' subject_id{sid} '_s']), f_surf);
    close(f_surf);
    f_surf2 = examinePathNode(Vertex_s, distance4D_s, path_node4D_s, xaxis, yaxis, ...
        xy2node_s, sxi, syi, txi, tyi, Face_s, Vertex_s, 0.2);
    screen2png(fullfile(saveDir, subject_id{sid}, ['surface_minimal_path_hmax' num2str(hmax) '_' subject_id{sid} '_s_transparent']), f_surf2);
    close(f_surf2);
    % %tst1 - now helping to reduce size
    % idx = find(~cellfun('isempty',path_node4D_s));
    % val = path_node4D_v(idx);
    % %tst2 - extremely slow
    % allvals = uint32(cat(1,path_node4D_s{:}));
    % offsets = cumsum([1; cellfun(@numel,path_node4D_s(:))]);


    %% save result
    tic
    saveName = fullfile(saveDir, subject_id{sid}, ['minimal_path_' type '_hmax' ...
        num2str(hmax) '_' subject_id{sid} '_s.mat']);
    saveName_tmp = fullfile(saveDir_tmp, 'tmp.mat');
    save(saveName_tmp,...
        'distance2D_s','xy2node_s','surfaceNodes_s',...
        'Face_s','Vertex_s','G_s');%);%,'-v7.3');%,'G')'path_node4D_s',
    t=toc
    movefile(saveName_tmp, saveName);
    t= toc %>10min



    %% %%%%%%%%%%% PATH MINIMIZATION IN BRAIN VOLUME %%%%%%%%%%%%%%%%

    %% 1. compute volumetric mesh
    tic;
    elements = model.Mesh.Elements';
    Vertex_v = model.Mesh.Nodes';

    % Initialize an empty array to store faces
    Face_v = [];
    % Loop through all elements to extract faces
    parfor i = 1:size(elements, 1)
        elem = elements(i, :);

        % Define faces of the current element
        elem_faces = [elem([1, 2, 3]);  % Face 1
            elem([1, 2, 4]);  % Face 2
            elem([1, 3, 4]);  % Face 3
            elem([2, 3, 4])]; % Face 4

        % Add faces to the faces list
        Face_v = [Face_v; elem_faces];
    end

    % Ensure unique faces (remove duplicate faces if any)
    %unique_faces = unique(sort(Face_v, 2), 'rows');
    t1=toc %~30s

    %% 2. convert volumetric mesh to graph
    tic
    % Assuming your FEMesh object has nodes and elements properties
    nodes = single(model.Mesh.Nodes');
    elements = single(model.Mesh.Elements');

    % Create an adjacency matrix based on element connectivity
    num_nodes = size(nodes, 1);

    % Create a graph object from edges and weights
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

    G_v = graph([sourcenode_all tgtnode_all], [tgtnode_all sourcenode_all], ...
        [distance_all distance_all]);
    G_v.Nodes.X = Vertex_v(:,1);
    G_v.Nodes.Y = Vertex_v(:,2);
    G_v.Nodes.Z = Vertex_v(:,3);
    t2 = toc

    %% 3. compute shortest path distance (volume)
    tic
    [distance4D_v, path_node4D_v, surfaceNodes_v, xy2node_v] ...
        = getDistance4D(G_v, grid_x, grid_y, array_3d);
    distance2D_v = reshape(distance4D_v, numel(yaxis)*numel(xaxis), numel(yaxis)*numel(xaxis));
    t3 = toc %~10min


    %% 4 sanity check
    f_vol = examinePathNode(Vertex_v, distance4D_v, path_node4D_v, xaxis, yaxis, ...
        xy2node_v, sxi, syi, txi, tyi, Face_s, Vertex_s, 0.3);
    screen2png(fullfile(saveDir, subject_id{sid}, ['surface_minimal_path_hmax' num2str(hmax) '_' subject_id{sid} '_v_transparent']), f_vol);
    close(f_vol);
    f_vol2 = examinePathNode(Vertex_v, distance4D_v, path_node4D_v, xaxis, yaxis, ...
        xy2node_v, sxi, syi, txi, tyi, Face_s, Vertex_s, 0.8);
    screen2png(fullfile(saveDir, subject_id{sid}, ['surface_minimal_path_hmax' num2str(hmax) '_' subject_id{sid} '_v']), f_vol2);
    close(f_vol2);


    %% save result
    saveName = fullfile(saveDir, subject_id{sid}, ['minimal_path_' type '_hmax' ...
        num2str(hmax) '_' subject_id{sid} '_v.mat']);
    saveName_tmp = fullfile(saveDir_tmp, 'tmp.mat');
    save(saveName_tmp,...
        'distance2D_v','xy2node_v','surfaceNodes_v',...
        'Face_v','Vertex_v','G_v');%);%,'-v7.3');%,'G')'path_node4D_s',
    t=toc
    movefile(saveName_tmp, saveName);
 
    close all
    clear G_s G_v distance2D_v distance2D_s surfaceNodes_v surfaceNodes_s ...
        Face_v Face_s Vertex_v Vertex_s
    catch err
        disp(['error in ' subject_id{sid}]);
        continue;
    end
end %sid

