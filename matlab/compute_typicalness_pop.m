server = '/mnt/dshi0006_market/VariabilityEarlyVisualCortex';

% thisSubject = '146432';%'585256';%'157336'; %typical
% thisSubject = '114823';%'725751';%'581450';%'114823'; %atypical
%getSubjectId;
parValues = [10 20 40 80 160];
tolerance = 45;%45;%deg
minNPix = 40;%30;%20; %number of pixels
R2th = 30;%
iroi = 3; %1:dorsal, 2:ventral, 3: both
showFig = 1;

% subjectId = {'105923','114823','789373','198653',...
%     '581450','397760','926862'}; %fernanda elife

% spearman correlation too high: 105923 198653 581450
% spearman correlation too low: 926862

subjectId = getSubjectId;

load(fullfile(server,  'avg', ['arealBorder_' 'avg']),...
    'grid_azimuth_i',"grid_altitude_i",'mask'); %make areaMatrix ??

[ecc_avg, pol_avg] = cart2pol(grid_azimuth_i, grid_altitude_i);

%% mask dorsal & ventral
load(fullfile(server, 'avg' ,['geometry_retinotopy_' 'avg' '.mat']),...
    'final_mask_L_d_idx');
roi_avg_d_tmp = zeros(100);
roi_avg_d_tmp(final_mask_L_d_idx) = 1;

if iroi == 1
    roi_avg = mask.*roi_avg_d_tmp;
elseif iroi == 2
    roi_avg = mask.*(1-roi_avg_d_tmp);
elseif iroi ==3
    roi_avg = mask;
end
    
pol_binary_avg = getPolarBinary(pol_avg, iroi, tolerance, minNPix, roi_avg);


for sid = 1:numel(subjectId)
    try
    thisSubject = subjectId{sid};
    %% image similarity
    load(fullfile(server, thisSubject ,['arealBorder_' thisSubject '.mat']),...
        'areaMatrix','grid_altitude_i','grid_azimuth_i','mask');

    %% mask dorsal & ventral
    load(fullfile(server, thisSubject ,['geometry_retinotopy_' thisSubject '.mat']),...
        'final_mask_L_d_idx','final_mask_L_idx');
    roi_d_tmp = zeros(100);
    roi_d_tmp(final_mask_L_d_idx) = 1;

    %% explained variance from Fernanda
    R2 = readNPY(fullfile(server, 'flatmaps',['R2_flatmap_' thisSubject '_LH.npy']));

    mask_v123 = areaMatrix{1}+areaMatrix{2}+areaMatrix{3};
    [ecc, pol] = cart2pol(grid_azimuth_i, grid_altitude_i);

    if iroi == 1
        roi = mask.*roi_d_tmp.*(rot90(R2,3) > R2th);
    elseif iroi == 2
        roi = mask.*(1-roi_d_tmp).* (rot90(R2,3) > R2th);
    elseif iroi ==3
        roi = mask.* (rot90(R2,3) > R2th);
    end
    
    tgtPixIdx = find(roi);

    pol_binary = getPolarBinary(pol, iroi, tolerance, minNPix, roi);

    similarity_e(sid,:,:) = polarSimilarityScore(pol, pol_avg, tolerance, minNPix, roi);
    corr_pol_e(sid) = corr(pol(tgtPixIdx), pol_avg(tgtPixIdx),'type','Spearman');
    %corr_pol_d = distcorr(pol(tgtPixIdx), pol_avg(tgtPixIdx));

    roi_rate(sid) = sum(roi(:))/sum(roi_avg(:))*100;

    if showFig
        figure;
        ax(1)=subplot(231);
        imagesc(pol_avg,'AlphaData',roi);
        axis square xy;
        set(gca,'xtick',[],'ytick',[]);
        colormap(ax(1),hsvr);
        clim([-180 180]);

        ax(2)=subplot(232);
        imagesc(pol,'AlphaData',roi);
        axis square xy;
        set(gca,'xtick',[],'ytick',[]);
        colormap(ax(2),hsvr);
        clim([-180 180]);
        title(['rank corr: ' num2str(corr_pol_e(sid))]);
        % title(sprintf('kendall: %.2f \n corrdist: %.2f',corr_pol_e, corr_pol_d));

        subplot(233);
        plot(pol_avg(find(roi)), pol(find(roi)),'.');
        axis square xy;
        squareplot;

        ax(3)=subplot(234);
        imagesc(pol_binary_avg,'AlphaData', roi);
        axis square xy;
        set(gca,'xtick',[],'ytick',[]);
        colormap(ax(3),"gray");

        ax(4)=subplot(235);
        imagesc(pol_binary,'AlphaData', roi);
        axis square xy;
        set(gca,'xtick',[],'ytick',[]);
        colormap(ax(4),"gray");
        title(['dice: ' num2str(similarity_e(sid,iroi,3))]);

        screen2png(['polarAngle_' thisSubject '_iroi' num2str(iroi)]);
        close;
    end

    catch err
    end
end

save(['typicalness_pop_roi' num2str(iroi)], 'similarity_e','corr_pol_e',...
    'roi_rate','tolerance', 'minNPix');

selected_index_final = intersect(selected_index, find(roi_rate>60));

histogram(corr_pol_e(selected_index_final),20);
vline(median(corr_pol_e(selected_index_final)));

[corr_pol_sorted, subjectId_sorted] = sort(corr_pol_e(selected_index_final));

subjectId(selected_index_final(subjectId_sorted))

    % [corr_pol_diff, subject_id_pol] = sort(corr_v_all(1,5,:,2)-corr_s_all(1,5,:,2));
    %
    % %% find maximum parameter pair for each subject
    % corrmax_v = nan(numel(subject_id),1);
    % corrmax_s = nan(numel(subject_id),1);
    % b1max_v = nan(numel(subject_id),1);
    % b2max_v = nan(numel(subject_id),1);
    % b1max_s = nan(numel(subject_id),1);
    % b2max_s = nan(numel(subject_id),1);
    %
    % ivda = 3;
    % isim=4;
    % for isub = 1:numel(subject_id)
    %     % thisMatrix = corr_v_all(:,:,isub,2);
    %     thisMatrix = similarity_v(:,:,isub,ivda,isim);
    %     [corrmax_v(isub),maxIdx]=max(thisMatrix(:));
    %     [b1max_v(isub),b2max_v(isub)]=ind2sub(size(thisMatrix),maxIdx);
    %
    %     % thisMatrix = corr_s_all(:,:,isub,2);
    %     thisMatrix = similarity_s(:,:,isub,ivda,isim);
    %     [corrmax_s(isub),maxIdx]=max(thisMatrix(:));
    %     [b1max_s(isub),b2max_s(isub)]=ind2sub(size(thisMatrix),maxIdx);
    % end
    %
    % load('subjectSelection.mat');
    % %plot(corrmax_s,corrmax_v,'.','Color',[.5 .5 .5]); hold on;
    % plot(corrmax_s(selected_index),corrmax_v(selected_index),'k.');
    % squareplot; marginplot;
    % xlabel('polar angle spearman correlation empirical v surface min');
    % ylabel('polar angle spearman correlation empirical v volume min');
    % close all;
% end