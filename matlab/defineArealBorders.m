%% defineArealBorders
% applies smoothing and thresholding on the field sign map
% allows specifying pixels by mouse clicking
% finds connected pixels 
% save smoothed vfs, sign of vfs and areaMatrix as  "arealBorder.mat" 
%
% this script requires imageProcessing toolbox (imfill, imgaussfilt)
%
% somehow worth result than the previous version ('fieldSign_'  subject_id{sid}   '_smoothed')??

subject_id = {'157336','585256','114823','581450','725751'}; %{'avg'};%{'146735','157336','585256','114823','581450','725751'};
loadDir = '/mnt/dshi0006_market/VariabilityEarlyVisualCortex/';%'/home/daisuke/Documents/git/VariabilityEarlyVisualCortex/results/';
saveDir = '/mnt/dshi0006_market/VariabilityEarlyVisualCortex/';

for sid = 1:length(subject_id)
    
    %% load field sign data
    %fname = ['fieldSign_'  subject_id{sid}   '_smoothed'];
    fname = ['geometry_retinotopy_' subject_id{sid}];
    load(fullfile(loadDir,  subject_id{sid}, [fname '.mat']), 'vfs');


    %% smoothing
    mask = ~isnan(vfs);
    vfs_c = interpNanImages(vfs);
    vfs_f = imgaussfilt(vfs_c, .5).*mask; %2
   % vfs_f = imdiffusefilt(vfs_c, 'GradientThreshold',10).*mask;
    %vfs_f = imbilatfilt(vfs_c, 2).*mask;
    

    %% thresholding
    threshold = .3;%.3;%th for binalizing vfs low > less space between borders
    std_signMap = nanstd(vfs_f(:));
	vfs_th = vfs_f;

    vfs_th(abs(vfs_th) < threshold*std_signMap) = 0;
	vfs_th(vfs_th >= threshold*std_signMap) = 1;
	vfs_th(vfs_th <= -threshold*std_signMap) = -1;

    figure;
    subplot(311); imagesc(vfs);colorbar; axis equal tight xy; title('original');
    subplot(312); imagesc(vfs_f);colorbar; axis equal tight xy; hold on;title('smoothed');
    subplot(313); imagesc(vfs_th);colorbar; axis equal tight xy;title('thresholded');
    linkaxes(findall(gcf, 'type', 'axes'))


    %% define areas by clicking region(s) of interest
    label{1} = 'V1';
    label{2} = 'V2';
    label{3} = 'V3';
    label{4} = 'V4';

    lcolor = lines(numel(label));

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
    screen2png(fullfile(saveDir, [fname, '_arealBorder.png']));


    %% save results
    save( fullfile(saveDir, [fname, '_arealBorder.mat']), ...
        'areaMatrix',"connectedPixels",'vfs_th',"vfs_f");
    close all;
end