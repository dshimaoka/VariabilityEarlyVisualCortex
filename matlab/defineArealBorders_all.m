%% defineArealBorders
% applies smoothing and thresholding on the field sign map
% allows specifying pixels by mouse clicking
% finds connected pixels 
%
% this script requires imageProcessing toolbox (imfill, imgaussfilt)

subject_id =  {'725751'}; %,'157336','585256','114823','581450','725751','avg'};
% saveDir = '/home/daisuke/Documents/git/VariabilityEarlyVisualCortex/results/';
saveDir = '/mnt/dshi0006_market/VariabilityEarlyVisualCortex/';
for sid = 1:length(subject_id)

    %% load field sign data
    retinotopyFilename = fullfile(saveDir,  subject_id{sid}, ['geometry_retinotopy_'  subject_id{sid}   '.mat']);
    load(retinotopyFilename, 'grid_azimuth','grid_altitude','grid_ecc','grid_PA') %'vfs', 'grid_ecc'
    grid_azimuth = pi/180 * grid_azimuth;
    grid_altitude = pi/180 * grid_altitude;
    mask = ~isnan(grid_altitude); %using a small mask is ciritical to obtain reasonable segmentation of vfs
   
    %% preprocessing retinotopy data (to be used for elastic net simulation)
     % interpolation
    th_retinotopy = 3;
    oddball = (abs(grid_azimuth) > th_retinotopy*std(grid_azimuth(mask))) | (abs(grid_altitude) > th_retinotopy*std(grid_altitude(mask)));
    interpolated = fillmissing2(grid_azimuth+1i*grid_altitude, 'natural','MissingLocations',oddball);
    grid_azimuth_i = real(interpolated);
    grid_altitude_i = imag(interpolated);

    
    %% define areal borders by Garrett 2014
    smoothingFac = 2;%3
    threshold = .3; %1.5
    [~,vfs_th, fig] = getHumanAreasX(180/pi*grid_azimuth, 180/pi*grid_altitude, smoothingFac, threshold, mask);
    screen2png(fullfile(saveDir, subject_id{sid},['areaSegmentation_' subject_id{sid} '_test.png']));
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
        areaMatrix{ii} = imfill(connectedMatrix{ii}.*mask==1, 'holes');
    end

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
    screen2png(fullfile(saveDir,  subject_id{sid}, ['arealBorder_' subject_id{sid}]));


    %% save results
    %save( [retinotopyFilename(1:end-4), '_arealBorder.mat'],...
    save(fullfile(saveDir,  subject_id{sid}, ['arealBorder_' subject_id{sid}]),... 
    'areaMatrix',"connectedPixels",'vfs_th');%,'-append'); %'vfs_f'
    close all;
end