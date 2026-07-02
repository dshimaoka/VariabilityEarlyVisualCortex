function [x,y,z,xaxis, yaxis, withinMask] = getXYZ(grid_x, grid_y, array_3d)
%[x,y,z,withinMask] = getXYZ(grid_x, grid_y, array_3d)
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