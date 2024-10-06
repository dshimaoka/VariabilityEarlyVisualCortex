function [mask, oddball_inner, oddball_outer] = getMask(grid_azimuth, grid_altitude, th_retinotopy)

    mask_ribeiro = ~isnan(grid_altitude); %generally this produces many small patches near mask border

    % interpolation using spatial derivative
    [dhdx, dhdy] = gradient(grid_azimuth);
    [dvdx, dvdy] = gradient(grid_altitude);
    tmp = (abs(dhdx)+abs(dhdy)+abs(dvdx)+abs(dvdy));
    oddball = abs(tmp) > th_retinotopy*std(tmp(~isnan(tmp))); 
   

    % define new mask
    SE = strel('disk', 2,0);
    oddball = imclose(oddball,SE);
    oddball = imfill(oddball, 'hole');
    oddball_all = imdilate(oddball,SE); %to include mask boundary
    
    [labeledImage, numComponents] = bwlabel(oddball_all | ~mask_ribeiro);
    labelsInOddball_all = unique(labeledImage(oddball_all));
    labels_outer = unique(labeledImage(~mask_ribeiro));
    labels_inner = setxor(labelsInOddball_all, labels_outer);

    oddball_inner = ismember(labeledImage, labels_inner);
    oddball_outer = ismember(labeledImage, labels_outer);
    mask_tmp = mask_ribeiro & ~oddball_outer; 

    [labeledImage, numComponents] = bwlabel(mask_tmp);
    stats = regionprops(labeledImage, 'Area');
    allAreas = [stats.Area];  % Extract areas of all components
    [~, largestComponentIdx] = max(allAreas);

    %Create a mask with only the largest component
    mask = (labeledImage == largestComponentIdx);