%% defineArealBorders
% applies smoothing and thresholding on the field sign map
% allows specifying pixels by mouse clicking
% finds connected pixels 
%
% this script requires imageProcessing toolbox (imfill, imgaussfilt)

subject_id =  {'157336','585256','114823','581450','725751','avg'};
% saveDir = '/home/daisuke/Documents/git/VariabilityEarlyVisualCortex/results/';
%ng mask 
% 725751: V1-V3 connected w threshold = .3
saveDir = '/mnt/dshi0006_market/VariabilityEarlyVisualCortex/';
for sid = [4 6]%1:length(subject_id)

    %% load field sign data
    retinotopyFilename = fullfile(saveDir,  subject_id{sid}, ['geometry_retinotopy_'  subject_id{sid}   '.mat']);
    load(retinotopyFilename, 'grid_azimuth','grid_altitude') %'vfs', 'grid_ecc'
    grid_azimuth = pi/180 * grid_azimuth;
    grid_altitude = pi/180 * grid_altitude;
    mask_ribeiro = ~isnan(grid_altitude); %generally this produces many small patches near mask border

    %% preprocessing retinotopy data (to be used for elastic net simulation)

    % interpolation using spatial derivative
    th_retinotopy = 2; %1;
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


     interpolated = fillmissing2(grid_azimuth+1i*grid_altitude, 'linear','MissingLocations',oddball_inner);
    grid_azimuth_i = real(interpolated);
    grid_altitude_i = imag(interpolated);

    %% define areal borders by Garrett 2014
    smoothingFac = 2;%3
    threshold = .3; %.3
    [~,vfs_th, vfs, fig] = getHumanAreasX(180/pi*grid_azimuth_i, 180/pi*grid_altitude_i, smoothingFac, threshold, mask);
    screen2png(fullfile(saveDir, subject_id{sid},['areaSegmentation_' subject_id{sid} '.png']));
    close all

    % figure;
        % subplot(411); imagesc(vfs);colorbar; axis equal tight xy; title('original');
        % subplot(412); imagesc(vfs_f);colorbar; axis equal tight xy; hold on;title('smoothed');
        % subplot(413); imagesc(vfs_th);colorbar; axis equal tight xy;title('thresholded');
        % subplot(414); imagesc(signBorder);colorbar; axis equal tight xy;title('Garret border');
        % linkaxes(findall(gcf, 'type', 'axes'))


    %% define areas by clicking region(s) of interest
    label{1} = 'V1';
    label{2} = 'V2';
    label{3} = 'V3';
    label{4} = 'V4';

    lcolor = lines(numel(label));

    imagesc(vfs);

    connectedPixels = [];
    for ii = 1:numel(label)
        figure; 
        [connectedPixels{ii}, connectedMatrix{ii}] = ...
            findConnectedPixels(vfs_th, label{ii});
        close;

        % fuse neighboring areas cf. fusePatchesX
        SE = strel('disk',1,0);
        connectedMatrix{ii} = imerode(imdilate(connectedMatrix{ii}, SE),SE);
        
        % fill holes
        areaMatrix{ii} = imfill(connectedMatrix{ii}.*mask==1, 'holes');
    end
    close all;

    % %for each pixel deemed as vfs_th=0, find nearest areas, assign it to
    % %one of the areas whos vfs values are closer
    % vfs_th(mask==0)=nan;
    % [zeros_r, zeros_c] = find(vfs_th==0);
    % zeros_sub = [zeros_c zeros_r];
    % 
    % [~, areaMatrix_new] = assignPixels(connectedPixels{ii}, areaMatrix{ii}.*vfs_th, vfs_f, zeros_sub);
    % 
    % vfs_th(areaMarix{ii})
% end


    %% visualization
    figure('position',[0 0 700 1500]);
    subplot(211);
    imagesc(vfs); colormap(gray); hold on
    for iarea = 1:numel(label)
        contour(areaMatrix{iarea},[.5 .5], 'edgecolor',lcolor(iarea,:),'linewidth',1);
        hold on;
    end
    axis xy equal;

    subplot(212);
    imagesc(vfs_th); colormap(gray); hold on
    for iarea = 1:numel(label)
        contour(areaMatrix{iarea}, [.5 .5], 'edgecolor',lcolor(iarea,:),'linewidth',3);
        hold on;
    end
    legend(label);
    axis xy equal;
    screen2png(fullfile(saveDir,  subject_id{sid}, ['arealLabels_' subject_id{sid}]));


    %% save results
    %save( [retinotopyFilename(1:end-4), '_arealBorder.mat'],...
    save(fullfile(saveDir,  subject_id{sid}, ['arealBorder_' subject_id{sid}]),... 
    'areaMatrix',"connectedPixels",'vfs_th');%,'-append'); %'vfs_f'
    close all;
end