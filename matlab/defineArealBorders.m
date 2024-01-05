%% defineArealBorders
% applies smoothing and thresholding on the field sign map
% allows specifying pixels by mouse clicking
% finds connected pixels 
%
% this script requires imageProcessing toolbox (imfill, imgaussfilt)

subject_id = {'avg'};%{'146735','157336','585256','114823','581450','725751'};
saveDir = '/home/daisuke/Documents/git/VariabilityEarlyVisualCortex/results/';

for sid = 1:length(subject_id)

    %% load field sign data
    vfsfilename = fullfile(saveDir, ['fieldSign_'  subject_id{sid}   '_smoothed.mat']);
    load(vfsfilename, 'vfs');


    %% smoothing
    mask = ~isnan(vfs);
    vfs_c = interpNanImages(vfs);
    vfs_f = imgaussfilt(vfs_c,2).*mask; 
    

    %% thresholding
    threshold = .3;%.1;%th for binalizing vfs low > less space between borders
    std_signMap = nanstd(vfs_f(:));
	vfs_th = vfs_f;

    vfs_th(abs(vfs_th) < threshold*std_signMap) = 0;
	vfs_th(vfs_th >= threshold*std_signMap) = 1;
	vfs_th(vfs_th <= -threshold*std_signMap) = -1;

    figure;
    subplot(311); imagesc(vfs);colorbar; axis equal tight xy; title('original');
    subplot(312); imagesc(vfs_f);colorbar; axis equal tight xy; hold on;title('smoothed');
    subplot(313); imagesc(vfs_th);colorbar; axis equal tight xy;title('thresholded');


    %% define areas by clicking region(s) of interest
    label{1} = 'V1';
    label{2} = 'V2';
    label{3} = 'V3';
    lcolor = lines(numel(label));

    connectedPixels = [];
    for ii = 1:3
        figure;
        [connectedPixels{ii}, connectedMatrix{ii}] = ...
            findConnectedPixels(vfs_th, label{ii});
        close;
        areaMatrix{ii} = imfill(connectedMatrix{ii}.*mask==1, 'holes');
    end


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
    screen2png([vfsfilename(1:end-4), '_arealBorder.png']);


    %% save results
    save( [vfsfilename(1:end-4), '_arealBorder.mat'],'areaMatrix',"connectedPixels",'vfs_th');
    close all;
end