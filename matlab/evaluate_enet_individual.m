server = '/mnt/dshi0006_market/VariabilityEarlyVisualCortex';

% thisSubject = '146432';%'585256';%'157336'; %typical
% thisSubject = '114823';%'725751';%'581450';%'114823'; %atypical
%getSubjectId;
parValues = [10 20 40 80 160];
tolerance = 45;%45;%deg
minNPix = 40;%30;%20; %number of pixels
iroi = 1; %1:dorsal, 2:ventral, 3: both

subjectId = {'105923','111312','114823','134829','148133','169747','181232',...
    '182739','191033','198653','199655','203418','212419','257845',...
    '385046','525541','581450','671855','706040','770352','789373','826353','973770'};
% subjectId = {'105923','114823','789373','198653',...
%     '581450','397760','926862'};
% spearman correlation too high: 105923 198653 581450
% spearman correlation too low: 926862

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

    %roi_similarity = mask .* (rot90(R2,3)>30);
    %roi_similarity = mask .* (pol > -90-tolerance) .* (pol < 90+tolerance);
    %tgtPixIdx = find(roi_similarity);
    %tgtPixIdx = union(find(areaMatrix{2}),find(areaMatrix{3}));
    %tgtPixIdx = find(areaMatrix{1});

    mask_v123 = areaMatrix{1}+areaMatrix{2}+areaMatrix{3};
    [ecc, pol] = cart2pol(grid_azimuth_i, grid_altitude_i);

    if iroi == 1
        roi = mask.*roi_d_tmp.*(areaMatrix{2}+areaMatrix{3});%.*(rot90(R2,3)>30);
    elseif iroi == 2
        roi = mask.*(1-roi_d_tmp).*(areaMatrix{2}+areaMatrix{3});%.* (rot90(R2,3)>30);
    elseif iroi ==3
        roi = mask.*(areaMatrix{2}+areaMatrix{3});%.* (rot90(R2,3)>30);
    end
    
    roiIdx = find(roi);

    pol_binary = getPolarBinary(pol, iroi, tolerance, minNPix, roi);

    similarity_e = polarSimilarityScore(pol, pol_avg, tolerance, minNPix, roi);
    corr_pol_e = corr(pol(roiIdx), pol_avg(roiIdx),'type','Kendall');%'Spearman');
    corr_pol_d = distcorr(pol(roiIdx), pol_avg(roiIdx));


    %% elastic net simulation
    for ib1 = 1:numel(parValues)
        b1 = parValues(ib1);

        for ib2 = 1:numel(parValues)
            b2 = parValues(ib2);

            load(fullfile(server, thisSubject, ['summary_V+D_' thisSubject '_b1_' num2str(b1) '_b2_' num2str(b2) '_vs.mat']), ...
                'result2d_v','result2d_s');

            [ecc_v, pol_v(:,:,ib2,ib1)] = cart2pol(result2d_v(:,:,1)', result2d_v(:,:,2)');
            [ecc_s, pol_s(:,:,ib2,ib1)] = cart2pol(result2d_s(:,:,1)', result2d_s(:,:,2)');

            %% image similarity
            similarity_v(ib2,ib1,:,:) = polarSimilarityScore(pol_v(:,:,ib2,ib1), pol, tolerance, minNPix, roi);
            similarity_s(ib2,ib1,:,:) = polarSimilarityScore(pol_s(:,:,ib2,ib1), pol, tolerance, minNPix, roi);

            pol_binary_v(:,:,ib2,ib1) = getPolarBinary(pol_v(:,:,ib2,ib1), iroi, tolerance, minNPix, roi);
            pol_binary_s(:,:,ib2,ib1) = getPolarBinary(pol_s(:,:,ib2,ib1), iroi, tolerance, minNPix, roi);

            %% correlation
            thispol_v = pol_v(:,:,ib2,ib1);
            thispol_s = pol_s(:,:,ib2,ib1);
            corr_pol_v(ib2,ib1) = corr(pol(roiIdx), thispol_v(roiIdx),'type','Spearman');
            corr_pol_s(ib2,ib1) = corr(pol(roiIdx), thispol_s(roiIdx),'type','Spearman');

            %% error
            error_pol_v(ib2,ib1) = median(abs(pol(roiIdx) - thispol_v(roiIdx)));
            error_pol_s(ib2,ib1) = median(abs(pol(roiIdx) - thispol_s(roiIdx)));
            
        end
    end


    % save('correlation_pop','similarity_s','similarity_v','minNPix','tolerance');

    stype=3; %dice
    %stype = 5; %turing dist
    colormap('parula');
    ax(1)=subplot(321);
    imagesc(squeeze(similarity_s(:,:,iroi,stype))');
    axis ij;
    squareplot;
    xlabel('b2');ylabel('b1');
    title('surface dice')

    ax(2)=subplot(322);
    imagesc(squeeze(similarity_v(:,:,iroi,stype))');
    axis ij;
    squareplot;
    xlabel('b2');ylabel('b1');
    linkcaxes(ax(1:2));
    mcolorbar;
    title('volume dice');

    ax(3)=subplot(323);
    imagesc(squeeze(corr_pol_s)');
    axis ij;
    squareplot;
    xlabel('b2');ylabel('b1');
    title('surface corr')

    ax(4)=subplot(324);
    imagesc(squeeze(corr_pol_v)');
    axis ij;
    squareplot;
    xlabel('b2');ylabel('b1');
    linkcaxes(ax(3:4));
    mcolorbar;
    title('volume corr');

    ax(5)=subplot(325);
    imagesc(squeeze(error_pol_s)');
    axis ij;
    squareplot;
    xlabel('b2');ylabel('b1');
    title('surface error')

    ax(6)=subplot(326);
    imagesc(squeeze(error_pol_v)');
    axis ij;
    squareplot;
    xlabel('b2');ylabel('b1');
    linkcaxes(ax(5:6));
    mcolorbar;
    title('volume error');

    screen2png(['dice_corr_error_' thisSubject '_iroi' num2str(iroi)]);
    close all;


    for isv = 1:2
        switch isv
            case 1
                theseImages = pol_binary_s;
                suffix = '_s';
            case 2
                theseImages = pol_binary_v;
                suffix = '_v';
        end
        figure('position',[0 0 1000 1200]);
        colormap("gray");
        subplot(numel(parValues)+1,numel(parValues),1);
        imagesc(squeeze(pol_binary));axis square xy;set(gca,'xtick',[],'ytick',[]);
        for ib1 = 1:numel(parValues) %y
            for ib2 = 1:numel(parValues) %x
                subplot(numel(parValues)+1,numel(parValues),numel(parValues)*ib1+ib2);
                imagesc(squeeze(theseImages(:,:,ib2,ib1)));axis square xy;
                set(gca,'xtick',[],'ytick',[]);
                if ib1==numel(parValues)
                    xlabel(['b2 ' num2str(parValues(ib2))]);
                end
                if ib2==1
                    ylabel(['b1 ' num2str(parValues(ib1))]);
                end
            end
        end
        screen2png(['polarAngle_binarised_' thisSubject suffix '_iroi' num2str(iroi)]);
        close;
    end


    %% polar angle
    if iroi==1
        mask_polar = mask_v123.*roi_d_tmp;
    elseif iroi == 2
        mask_polar = mask_v123.*(1-roi_d_tmp);
    elseif iroi == 3
        mask_polar = mask_v123;
    end

    for isv = 1:2
        switch isv
            case 1
                theseImages = pol_s;
                suffix = '_s';
            case 2
                theseImages = pol_v;
                suffix = '_v';
        end
        figure('position',[0 0 1000 1200]);
        colormap(hsvr);
        subplot(numel(parValues)+1,numel(parValues),1);
        imagesc(squeeze(pol),'AlphaData',mask_polar);axis square xy;set(gca,'xtick',[],'ytick',[]);clim([-180 180]);
        for ib1 = 1:numel(parValues) %y
            for ib2 = 1:numel(parValues) %x
                subplot(numel(parValues)+1,numel(parValues),numel(parValues)*ib1+ib2);
                imagesc(squeeze(theseImages(:,:,ib2,ib1)),'AlphaData',mask_polar);
                axis square xy;
                set(gca,'xtick',[],'ytick',[]);
                clim([-180 180]);
                if ib1==numel(parValues)
                    xlabel(['b2 ' num2str(parValues(ib2))]);
                end
                if ib2==1
                    ylabel(['b1 ' num2str(parValues(ib1))]);
                end
            end
        end
        screen2png(['polarAngle_' thisSubject suffix '_iroi' num2str(iroi)]);
        close;
    end

    %
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
    close all;
end