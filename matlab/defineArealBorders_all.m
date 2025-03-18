%% defineArealBorders
% applies smoothing and thresholding on the field sign map
% allows specifying pixels by mouse clicking
% finds connected pixels
%
% this script requires imageProcessing toolbox (imfill, imgaussfilt)

%subject_id =  {'157336','585256','114823','581450','725751','avg'};
%subject_id = getSubjectId;
subject_id = {'169343','169344'};

%ng
saveDir = '/mnt/dshi0006_market/VariabilityEarlyVisualCortex/';

th_retinotopy = 3.5; %1; %threshold for spatial gradient
threshold_vfs = .5;%0.3 default threshold for vfs
smoothingFac = 2;%3 %smoothing in Garrett 2014

ngIdx = [];
for sid = 2:length(subject_id)

    try
        disp(subject_id(sid));

        %% load field sign data
        retinotopyFilename = fullfile(saveDir,  subject_id{sid}, ['geometry_retinotopy_'  subject_id{sid}   '.mat']);
        load(retinotopyFilename, 'grid_azimuth','grid_altitude') %'vfs', 'grid_ecc'

        % get mask within which retinotopy gradient is smooth
        [mask, oddball_inner, oddball_outer] = getMask(pi/180 *grid_azimuth, pi/180 *grid_altitude, ...
            th_retinotopy);

        % interpolate pixels with odd values inside the mask
        interpolated = fillmissing2(pi/180 *grid_azimuth+1i*pi/180 *grid_altitude, 'linear','MissingLocations',oddball_inner);
        grid_azimuth_i = 180/pi*real(interpolated); %[deg]
        grid_altitude_i = 180/pi*imag(interpolated); %[deg]

        %% define areal borders by Garrett 2014
        if any(strcmp(subject_id{sid},{'585256'}))
            threshold = .7;
        elseif any(strcmp(subject_id{sid},{'725751'}))
            threshold = .5;
        else
            threshold = threshold_vfs;
        end

        [~,vfs_th, vfs_f, fig] = getHumanAreasX(grid_azimuth_i, grid_altitude_i, ...
            smoothingFac, threshold, mask);
        screen2png(fullfile(saveDir, subject_id{sid},['areaSegmentation_' subject_id{sid} '.png']));
        close all

        %% define areas by clicking region(s) of interest
        label{1} = 'V1';
        label{2} = 'V2';
        label{3} = 'V3';
        label{4} = 'V4';

        lcolor = lines(numel(label));

        figure('position',[1400          1         500         500])
        imagesc(vfs_f); axis xy square tight;

        connectedPixels = [];
        for ii = 1:numel(label)
            figure('position',[1400          500         500         500]);
            [connectedPixels{ii}, connectedMatrix{ii}] = ...
                findConnectedPixels(vfs_th, label{ii},[],true);
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
        imagesc(vfs_f); colormap(gray); hold on
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
            'areaMatrix',"connectedPixels",'vfs_th','vfs_f','grid_azimuth_i',"grid_altitude_i",'threshold',...
            'smoothingFac','mask');
        close all;
    catch err
        ngIdx = [ngIdx subject_id(sid)];
        close all
    end
end