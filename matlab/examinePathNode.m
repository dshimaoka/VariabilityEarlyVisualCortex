function f = examinePathNode(vertices, distance4D, path_node4D, xaxis, yaxis, ...
    xy2node, sxi, syi, txi, tyi, faces_s, vertices_s, transparency)
% f = examinePathNode(vertices, distance4D, path_node4D, xaxis, yaxis, ...
%     xy2node, sxi, syi, txi, tyi, faces_s, vertices_s, transparency)
% creats a figure to check the path_node in a 3d volume
% INPUT
% faces_s, vertices_s, transparency: brain surface data 
% sxi, syi: source nodes (N x 1)
% txi, tyi: target nodes (N x 1)

if ~isequal(size(sxi), size(syi), size(txi), size(tyi))
    error('sxi, syi, txi, and tyi must have the same dimensions.');
end

% nodes = G.Nodes
%transparency = .8;%0.3; %0: clear 
facecolor = [.7 .7 .7];
% margin = 10;

% vidx_s = xy2node(syi,sxi);%source voxel
vidx_s = [];
vidx_t = []; %target voxel
for ii = 1:numel(txi)
    vidx_s(ii) = xy2node(syi(ii), sxi(ii));%source voxel
    vidx_t(ii) = xy2node(tyi(ii), txi(ii)); %flat2vert(tyi(ii),txi(ii));
end


% xrange = prctile(vertices([vidx_s vidx_t], 1),[0 100]) + [-margin margin];
% yrange = prctile(vertices([vidx_s vidx_t], 2),[0 100]) + [-margin margin];
% zrange = prctile(vertices([vidx_s vidx_t], 3),[0 100]) + [-margin margin];
xrange = [-50 10];
yrange = [-90 -30]-20;
zrange = [-50 10]+20;

f = figure('position',[0 0 1980 1080]);
subplot(141); %distance on flattened map from (sx,sy)
imagesc(xaxis, yaxis, squeeze(distance4D(syi(1),sxi(1),:,:)));hold on;
scatter(xaxis(sxi(1)), yaxis(syi(1)), 20, 'filled', 'MarkerFaceColor', 'm');
for ii = 1:numel(txi)
    scatter(xaxis(txi(ii)), yaxis(tyi(ii)), 20, 'filled', 'MarkerFaceColor', 'g');
    lc = [ii/numel(txi) 0 0];
    line([xaxis(sxi(ii)) xaxis(txi(ii))], [yaxis(syi(ii)) yaxis(tyi(ii))],'color',lc);
end
xlabel('x'); ylabel('y');
title('shortest path distance');
colorbar;
axis equal tight xy; grid on;

for vv = 1:3
    ax(vv) = subplot(1,4,vv+1); %minimal path between (sx,sy) and (tx,ty)
    % trisurf(mid_gifti_L.faces, mid_gifti_L.vertices(:,1),...
    %     mid_gifti_L.vertices(:,2),mid_gifti_L.vertices(:,3),'FaceColor', ...
    %     facecolor, 'EdgeColor', 'none','facealpha',transparency);
    trisurf(faces_s, vertices_s(:,1), vertices_s(:,2), vertices_s(:,3), 'FaceColor', ...
        facecolor, 'EdgeColor', 'none','facealpha',transparency); hold on;
    hold on;
    for ii = 1:numel(txi)
        scatter3(vertices(vidx_s, 1), vertices(vidx_s, 2), vertices(vidx_s, 3), 20, 'm', 'filled');
        scatter3(vertices(vidx_t(ii), 1), vertices(vidx_t(ii), 2), vertices(vidx_t(ii), 3), 20, 'green', 'filled');
    end
    for ii = 1:numel(txi)
        lc = [ii/numel(txi) 0 0];
        plot3(vertices(path_node4D{syi(ii),sxi(ii),tyi(ii),txi(ii)}, 1), ...
            vertices(path_node4D{syi(ii),sxi(ii),tyi(ii),txi(ii)}, 2), ...
            vertices(path_node4D{syi(ii),sxi(ii),tyi(ii),txi(ii)}, 3), '-', 'color',lc,'LineWidth', 2);
    end
    set(gca,'view',[30 10]);
    xlabel('x'); ylabel('y'); zlabel('z');
    axis equal tight;
    xlim(xrange);    ylim(yrange);    zlim(zrange);
    hold off;

    camlight('headlight');
    %lighting gouraud;

    % 4. Adjust surface reflectance properties
    material('shiny');

    switch vv
        case 1
            set(ax(vv),'view',[0 90]);
        case 2
            set(ax(vv),'view',[90 0]);
        case 3
            set(ax(vv),'view',[0 0]);
    end
end