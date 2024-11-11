function theta_binary_masked = getBinaryTheta(theta, tolerance, minNPix, mask)
%theta: [deg]

if nargin < 4
    mask = ~isnan(theta);

    %all areas including V1
    %mask = original.mask;

    %V2+V3
    % mask = imclose(original.areaMatrix{2}+original.areaMatrix{3}, strel('disk',1)); %connect areal boundaries
end

%tolerance = 45;%deg
%minNPix = 20; %minimum number of pixels to retain 

% house theta in [0 360] deg
idx = theta < 0;
theta(idx) = theta(idx) + 360;

theta_ori_binary = zeros(size(theta));
theta_ori_binary(theta<90+tolerance & theta>90-tolerance) = 1;

theta_binary_masked = theta_ori_binary .* mask;

%remove small dots
theta_binary_masked = bwareafilt(logical(theta_binary_masked), [minNPix, Inf]);
