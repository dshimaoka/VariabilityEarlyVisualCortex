server = '/mnt/dshi0006_market/VariabilityEarlyVisualCortex';
subject_id = {'105923','111312','114823','134829','148133','169747','181232',...
    '182739','191033','198653','199655','203418','212419','257845',...
    '385046','525541','581450','671855','706040','770352','789373','826353','973770'};
% getSubjectId;
parValues = [10 20 40 80 160];
iroi = 1;

%% image similarity
tolerance = 45;%45;%deg
minNPix = 40;%30;%20; %number of pixels


error_id = {};
corr_v = nan(numel(parValues),numel(parValues),numel(subject_id),2);
corr_s = nan(numel(parValues),numel(parValues),numel(subject_id),2);
error_v = nan(numel(parValues),numel(parValues),numel(subject_id));
error_s = nan(numel(parValues),numel(parValues),numel(subject_id));
for isub = 1:numel(subject_id)
    thisSubject = subject_id{isub};
    disp(thisSubject);
    try
        load(fullfile(server, thisSubject ,['arealBorder_' thisSubject '.mat']),...
            'areaMatrix','grid_altitude_i','grid_azimuth_i','mask');

        load(fullfile(server, thisSubject ,['geometry_retinotopy_' thisSubject '.mat']),...
        'final_mask_L_d_idx','final_mask_L_idx');
        roi_d_tmp = zeros(100);
        roi_d_tmp(final_mask_L_d_idx) = 1;

        for ib1 = 1:numel(parValues)
            b1 = parValues(ib1);

            for ib2 = 1:numel(parValues)
            b2 = parValues(ib2);

                load(fullfile(server, thisSubject, ['summary_V+D_' thisSubject '_b1_' num2str(b1) '_b2_' num2str(b2) '_vs.mat']), ...
                    'result2d_v','result2d_s');

                if iroi == 1
                    roi = roi_d_tmp.*(areaMatrix{2}+areaMatrix{3});
                elseif iroi == 2
                    roi = (1-roi_d_tmp).* (areaMatrix{2}+areaMatrix{3});
                elseif iroi ==3
                    roi = (areaMatrix{2}+areaMatrix{3});
                end

                roiIdx = find(roi);%union(find(areaMatrix{2}),find(areaMatrix{3}));
                %tgtPixIdx = find(areaMatrix{1});

                [ecc, pol] = cart2pol(grid_azimuth_i, grid_altitude_i);
                [ecc_v, pol_v(:,:,ib2,ib1)] = cart2pol(result2d_v(:,:,1)', result2d_v(:,:,2)');
                [ecc_s, pol_s(:,:,ib2,ib1)] = cart2pol(result2d_s(:,:,1)', result2d_s(:,:,2)');

                %% image similarity
                similarity_v(ib2,ib1,isub,:,:) = polarSimilarityScore(pol_v(:,:,ib2,ib1), pol, tolerance, minNPix, mask);
                similarity_s(ib2,ib1,isub,:,:) = polarSimilarityScore(pol_s(:,:,ib2,ib1), pol, tolerance, minNPix, mask);

                pol_binary = getBinaryTheta(interpNanImages(-pol), tolerance, minNPix, mask) ...
                    + getBinaryTheta(interpNanImages(pol), tolerance, minNPix, mask);

                pol_binary_v(:,:,ib2,ib1) = getBinaryTheta(interpNanImages(-pol_v(:,:,ib2,ib1)), tolerance, minNPix, mask) ...
                    + getBinaryTheta(interpNanImages(pol_v(:,:,ib2,ib1)), tolerance, minNPix, mask);
                pol_binary_s(:,:,ib2,ib1) = getBinaryTheta(interpNanImages(-pol_s(:,:,ib2,ib1)), tolerance, minNPix, mask) ...
                    + getBinaryTheta(interpNanImages(pol_s(:,:,ib2,ib1)), tolerance, minNPix, mask);

                %% correlation
                thispol_v = pol_v(:,:,ib2,ib1);
                thispol_s = pol_s(:,:,ib2,ib1);

                corr_v(ib2,ib1,isub) = corr(pol(roiIdx), thispol_v(roiIdx),'type','Spearman');
                corr_s(ib2,ib1,isub) = corr(pol(roiIdx), thispol_s(roiIdx),'type','Spearman');

                %% error
                error_v(ib2,ib1,isub) = median(abs(pol(roiIdx) - thispol_v(roiIdx)));
                error_s(ib2,ib1,isub) = median(abs(pol(roiIdx) - thispol_s(roiIdx)));
            end
        end

        thisMatrix = error_v(:,:,isub);
        [errormin_v(isub),maxIdx]=min(thisMatrix(:));
        [b2min_v(isub),b1min_v(isub)]=ind2sub(size(thisMatrix),maxIdx);

        thisMatrix = error_s(:,:,isub);
        [errormin_s(isub),maxIdx]=min(thisMatrix(:));
        [b2min_s(isub),b1min_s(isub)]=ind2sub(size(thisMatrix),maxIdx);

        %% compute if the error values are systemically different between volume and surface simulations
        bestpol_v = pol_v(:,:,b2min_v(isub), b1min_v(isub));
        bestpol_s = pol_s(:,:,b2min_s(isub), b1min_s(isub));
        [~,~,~,stats] = ttest(abs(pol(roiIdx) - bestpol_v(roiIdx)),abs(pol(roiIdx) - bestpol_s(roiIdx)));

        tstat(isub) = stats.tstat; %<0: volume excels, >0: surface excels

        a(1)=subplot(221);imagesc(bestpol_s,'alphadata',roi);clim([-180 180]); colormap(a(1),hsvr);
        a(2)=subplot(222);imagesc(bestpol_v,'alphadata',roi);clim([-180 180]); colormap(a(2),hsvr);
        a(3)=subplot(223);imagesc(abs(pol-bestpol_s),'alphadata',roi);clim([0 40]); colormap(a(3),parula);
        a(4)=subplot(224);imagesc(abs(pol-bestpol_v),'alphadata',roi);clim([0 40]); colormap(a(4),parula);
    catch err
        disp(err);
        error_id{numel(error_id)+1} = thisSubject;
    end
end
 
CMOsave(['evaluate_enet_pop_roi' num2str(iroi)],'similarity_s','similarity_v','minNPix','tolerance',...
    "corr_v",'corr_s','error_s','error_v');

load('subjectSelection.mat','selected_index');
load(['typicalness_pop_roi' num2str(iroi)],'roi_rate','corr_pol_e');

selected_index_final = intersect(selected_index, find(roi_rate>60));

stype=3; %dice
colormap('parula');
ax(1)=subplot(321);
imagesc(squeeze(nanmean(similarity_s(:,:,selected_index_final,iroi,stype),3))');
axis ij;
squareplot;
xlabel('b2');ylabel('b1');
title('surface dice')

ax(2)=subplot(322);
imagesc(squeeze(nanmean(similarity_v(:,:,selected_index_final,iroi,stype),3))');
axis ij;
squareplot;
xlabel('b2');ylabel('b1');
linkcaxes(ax(1:2));
mcolorbar;
title('volume dice');

ax(3)=subplot(323);
imagesc(squeeze(nanmean(corr_s(:,:,selected_index_final,1),3))');
axis ij;
squareplot;
xlabel('b2');ylabel('b1');
title('surface corr')

ax(4)=subplot(324);
imagesc(squeeze(nanmean(corr_v(:,:,selected_index_final,1),3))');
axis ij;
squareplot;
xlabel('b2');ylabel('b1');
linkcaxes(ax(3:4));
mcolorbar;
title('volume corr');

ax(5)=subplot(325);
imagesc(nanmean(error_s(:,:,selected_index_final),3)');
axis ij;
squareplot;
xlabel('b2');ylabel('b1');
title('surface error')

ax(6)=subplot(326);
imagesc(nanmean(error_v(:,:,selected_index_final),3)');
axis ij;
squareplot;
xlabel('b2');ylabel('b1');
linkcaxes(ax(5:6));
mcolorbar;
title('volume error');

screen2png(['dice_corr_error_pop_iroi' num2str(iroi)]);
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
    screen2png(['polarAngle_binarised_' subject_id{1} suffix]);
    close;
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
    colormap("hsv");
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
    screen2png(['polarAngle_binarised_' subject_id{1} suffix]);
    close;
end

 
%% find parameter pair w minimum error for each subject
errormin_v = nan(numel(subject_id),1);
errormin_s = nan(numel(subject_id),1);
b1min_v = nan(numel(subject_id),1);
b2min_v = nan(numel(subject_id),1);
b1min_s = nan(numel(subject_id),1);
b2min_s = nan(numel(subject_id),1);

for isub = 1:numel(subject_id)
    thisMatrix = error_v(:,:,isub);
    [errormin_v(isub),maxIdx]=min(thisMatrix(:));
    [b2min_v(isub),b1min_v(isub)]=ind2sub(size(thisMatrix),maxIdx);

    thisMatrix = error_s(:,:,isub);
    [errormin_s(isub),maxIdx]=min(thisMatrix(:));
    [b2min_s(isub),b1min_s(isub)]=ind2sub(size(thisMatrix),maxIdx);

    %% compute if the error values are systemically different between volume and surface simulations

end


figure('position',[0 0 1000 500])
subplot(121);
histogram2(b2min_s(selected_index_final), b1min_s(selected_index_final),.5:5.5,.5:5.5,'DisplayStyle','tile');
set(gca,'xticklabel',parValues,'YTickLabel',parValues);
xlabel('b2');ylabel('b1');axis ij; squareplot;mcolorbar;
title('minimize error for surface simulation')

subplot(122);
histogram2(b2min_v(selected_index_final), b1min_v(selected_index_final),.5:5.5,.5:5.5,'DisplayStyle','tile');
set(gca,'xticklabel',parValues,'YTickLabel',parValues);
xlabel('b2');ylabel('b1');axis ij; squareplot;mcolorbar;
title('minimize error for volume simulation')


figure;
th = median(corr_pol_e(selected_index_final));
subplot(121);
plot(corr_pol_e(selected_index_final),errormin_v(selected_index_final)-errormin_s(selected_index_final),'.')
hline(0);vline(th,gca,'-','r');
xlabel('spearman correlation to gnd avg');
ylabel('error volume - error surface');

subject_index_t = intersect(find(corr_pol_e>th), selected_index_final);
subject_index_a = intersect(find(corr_pol_e<th), selected_index_final);

eaxis = -10:1:10;%5:3:40;
ax(1)=subplot(222);
histogram(errormin_v(subject_index_t)-errormin_s(subject_index_t),eaxis);
title(['corr > ' num2str(th)]);

ax(2)=subplot(224);
histogram(errormin_v(subject_index_a)-errormin_s(subject_index_a),eaxis);
xlabel('error volume - error surface');ylabel('#subjects');
title(['corr < ' num2str(th)]);
linkaxes(ax(1:2),'x');

