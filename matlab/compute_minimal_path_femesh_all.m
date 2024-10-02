subject_id = {'avg','114823','157336','585256','581450','725751'};
%114823: meshing failed for hmax of 2 and hmin of 1

%loadDir = '/home/daisuke/Documents/git/VariabilityEarlyVisualCortex/data/';
saveDir = '/mnt/dshi0006_market/VariabilityEarlyVisualCortex/';

type = 'midthickness';%'white' %cannot generateMesh with 'pial'
hmax = 2; %1: fine but too slow, 3: too coarse

for sid= 1:numel(subject_id) 
    % from export_geometry_individual.py
    load(fullfile(saveDir, subject_id{sid}, ['geometry_retinotopy_'  subject_id{sid}   '.mat']),...
        'array_3d','grid_x','grid_y', 'grid_curv');%,'mask');
    
    % from defineArealBorders_individual.m
    load(fullfile(saveDir,subject_id{sid},['arealBorder_' subject_id{sid}]),...
        'areaMatrix');
     roi = (areaMatrix{1}+areaMatrix{2}+areaMatrix{3})>0;

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
    %figure;plot3(x,y,z,'.');xlabel('x');ylabel('y');zlabel('z');


    %% 1. import entire brain
    tic
    model = createpde(1);
    importGeometry(model, fullfile(saveDir, subject_id{sid}, ['Geom_'  subject_id{sid} '_hclaplacian.stl'])); %"BracketTwoHoles.stl");%
    %pdegplot(model)
    generateMesh(model,"Hmax",hmax);%,"geometricOrder","linear","Hmin",0.2*mm); %determines coarseness of the mesh

   t1=toc %20s

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
    t2=toc %~30s

    %% 3. convert FEMesh to graph
    tic
    % Assuming your FEMesh object has nodes and elements properties
    nodes = model.Mesh.Nodes';
    elements = model.Mesh.Elements';

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
    close all

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


    save(fullfile(saveDir, subject_id{sid}, ['minimal_path_' type '_hmax' num2str(hmax) '_' subject_id{sid}]),...
        'distance4D','distance2D','xy2node','surfaceNodes',...
        'distance4D_euc','distance2D_euc');%,'G','-v7.3');


  %% show surface and shortest path distance
    %xidx = 54; yidx = 63; %61
    xidx = 44; yidx=63;
    figure('position',[0 0 1900 1000]);

    ax(3)=subplot(133);
    imagesc(xaxis, yaxis, squeeze(distance4D(yidx,xidx,:,:)));hold on;
    scatter(xaxis(xidx), yaxis(yidx), 50, 'filled', 'MarkerFaceColor', 'r');
    xlabel('x'); ylabel('y');
    title('shortest path distance');
    colorbar;
    axis equal tight xy; grid on;
    clim([0 20])

    ax(1)=subplot(1,3,1);
    trisurf(faces, vertices(:,1), vertices(:,2), vertices(:,3), 'FaceColor', 'c', 'EdgeColor', 'k');%,'facealpha', .05);
    shading interp
    light
    material dull
    hold on;
    % scatter3(vertices(find(mask), 1), vertices(find(mask), 2), vertices(find(mask), 3), 10, 'b', 'filled');
    scatter3(vertices(xy2node(yidx,xidx), 1), vertices(xy2node(yidx,xidx), 2), vertices(xy2node(yidx,xidx), 3), 300, 'r', 'filled');
    view(90, 0);
    title('surface from midline');
    axis ij equal tight off;

    ax(2)=subplot(1,3,2);
    imagesc(xaxis, yaxis, grid_curv);hold on;
    xlabel('x'); ylabel('y');
    title('curvature');
    colorbar;
    axis equal tight xy; grid on;
 
    
    linkaxes(ax([2 3]));

     screen2png(fullfile(saveDir, subject_id{sid}, ['surface_minimal_path_hmax' num2str(hmax) '_' subject_id{sid}]));


    %% sanity check 2

 figure('position',[0 0 1980 1080]);
 for ii = 1:6
     
     % sy = 25+1*ii;sx=70+1;
    sy =50 + 2*ii; sx = 43 +2*ii;


     distance4D_tmp = squeeze(distance4D_euc(sy,sx,:,:));
     maskV1 = areaMatrix{1}.*1;
     maskV1(maskV1==0)=nan;
     maskV2 = areaMatrix{2}.*1;
     maskV2(maskV2==0)=nan;
     distanceV1 = distance4D_tmp .* maskV1;
     %option1: minimum
     [~,minidx] = nanmin(distanceV1(:),[],1);
     [ty,tx] = ind2sub([size(distance4D,1) size(distance4D,2)], minidx);
     %option2: CoM
     distanceV1_c = 1./exp(distanceV1);
     distanceV1_c(isnan(distanceV1_c)) = 0;
     props = regionprops(true(size(distanceV1_c)), distanceV1_c, 'WeightedCentroid');
     ty1c = props.WeightedCentroid(2); tx1c = props.WeightedCentroid(1);

     distanceV2 = distance4D_tmp .* maskV2;
     [~,minidx] = nanmin(distanceV2(:),[],1);
     [ty2,tx2] = ind2sub([size(distance4D,1) size(distance4D,2)], minidx);

     ax(ii)=subplot(1,6,ii);
     imagesc(distance4D_tmp ,'AlphaData',roi);hold on;
     plot(sx,sy,'rx', tx,ty,'ro', tx1c,ty1c,'yo');
     axis equal tight xy; grid on; hold off;
     % xlim([70 90]);ylim([20 40]);
     if ii==1
         title('distance from source (x)')
         legend('source','tgt min distance','tgt CoM')
     end
 end
 linkaxes(ax);

 screen2png(fullfile(saveDir,subject_id{sid},['minimal_path_' type '_hmax' num2str(hmax) '_serie_' subject_id{sid}]))


  
    %
    % % Set the range of rotation angles
    % angles = -90:2:90;  % Change the increment (5 degrees) as needed
    %
    % % Initialize GIF file
    % filename = 'rotating_3d_figure.gif';
    %
    % % Loop through each angle and capture frame for the GIF
    % for i = 1:length(angles)
    %     view([angles(i), 0]);
    %
    %     % Capture the current figure as an image
    %     frame = getframe(gcf);
    %     im = frame2im(frame);
    %     [imind, cm] = rgb2ind(im, 256);
    %
    %     % Write to the GIF File
    %     if i == 1
    %         imwrite(imind, cm, filename, 'gif', 'Loopcount', inf, 'DelayTime', 0.1);
    %     else
    %         imwrite(imind, cm, filename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
    %     end
    % end
    %
    %
close all
end %sid

