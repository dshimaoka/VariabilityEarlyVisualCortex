%subject_id = {'115017','898176','346137','910241','958976','654552'};
% subject_id = {'463040','114823'};%
subject_id = getSubjectId;

%ng
saveDir = '/mnt/dshi0006_market/VariabilityEarlyVisualCortex/';

%th_retinotopy = 3.5; %1; %threshold for spatial gradient
%threshold_vfs = .5;%0.3 default threshold for vfs
%smoothingFac = 2;%3 %smoothing in Garrett 2014

%% binarised polar map - from showParameterSpace_similarity_test.m
tolerance = 30;%45;%deg
minNPix = 30;%20; %number of pixels

%% load ground average data
load(fullfile(saveDir,  'avg', ['arealBorder_' 'avg']),...
    'grid_azimuth_i',"grid_altitude_i",'mask');
mask_avg = mask;
[ecc_avg, pol_avg] = cart2pol(grid_azimuth_i, grid_altitude_i);

for itype = 1:3
    switch itype
        case 1
            pol_binary_avg(:,:,itype) = getBinaryTheta(interpNanImages(-pol_avg), tolerance, minNPix, mask);
        case 2
            pol_binary_avg(:,:,itype) = getBinaryTheta(interpNanImages(pol_avg), tolerance, minNPix, mask);
        case 3
            pol_binary_avg(:,:,itype) = logical(getBinaryTheta(interpNanImages(-pol_avg), tolerance, minNPix, mask) ...
                + getBinaryTheta(interpNanImages(pol_avg), tolerance, minNPix, mask));
    end
    stats = regionprops(pol_binary_avg(:,:,itype), 'ConvexHull');
    pol_poly_avg(:,:,itype) = polyshape(stats(1).ConvexHull);
end


overlap = nan(numel(subject_id),1);
coverage_v1 = nan(numel(subject_id),1);
coverage_v2v3 = nan(numel(subject_id),1);
corr_ecc = nan(numel(subject_id),1);
corr_pol = nan(numel(subject_id),1);
score_b =  nan(numel(subject_id),1);
score_c =  nan(numel(subject_id),1);
score_d =  nan(numel(subject_id),1);
score_j =  nan(numel(subject_id),1);
score_t = nan(numel(subject_id),1);
for sid = 1:numel(subject_id)

    try
        load(fullfile(saveDir,  subject_id{sid}, ['arealBorder_' subject_id{sid}]),...
            'areaMatrix','grid_azimuth_i',"grid_altitude_i",'mask');


        %% compute overlap of visual field coverage between V1 and V2+V3
        im = areaMatrix{1}+areaMatrix{2}+areaMatrix{3}+areaMatrix{4};

        kmap_hor = grid_azimuth_i;
        kmap_vert = grid_altitude_i;
        kmap_hor(~mask) = 0;
        kmap_vert(~mask) = 0;
        pixpermm = 1;

        [spCov_v1, spAxis, coverage_v1(sid)] = getVisFieldCoverage(areaMatrix{1},kmap_hor,kmap_vert,pixpermm);
        [spCov_v2v3, spAxis, coverage_v2v3(sid)] = getVisFieldCoverage(areaMatrix{2}+areaMatrix{3},kmap_hor,kmap_vert,pixpermm);


        overlap(sid) = sum(spCov_v2v3(:)+spCov_v1(:)==2) / max(sum(spCov_v1(:)), sum(spCov_v2v3(:))) * 100; %[percentage]
        disp(['Subject: ' subject_id{sid}]);
        disp(['Overlap: ' num2str(overlap(sid)) '[%]']);
        disp(['V1 Coverage: ' num2str(coverage_v1(sid)) '[deg^2]']); %[deg^2]
        disp(['V2/V3 Coverage: ' num2str(coverage_v2v3(sid)) '[deg^2]']); %[deg^2];



        %% compute spearman correlation of retinotopy to ground average
        tgtPixIdx = intersect(find(mask_avg), find(mask));
        [ecc, pol] = cart2pol(grid_azimuth_i, grid_altitude_i);
        corr_ecc(sid) = corr(ecc_avg(tgtPixIdx), ecc(tgtPixIdx),'type','Spearman');
        corr_pol(sid) = corr(pol_avg(tgtPixIdx), pol(tgtPixIdx),'type','Spearman');

         %% image similarity
        similarity(sid,:,:) = polarSimilarityScore(pol, pol_avg, tolerance, minNPix, mask);
        %    for itype = 1:3
        %     switch itype
        %         case 1
        %             pol_binary = getBinaryTheta(interpNanImages(-pol), tolerance, minNPix, mask);
        %         case 2
        %             pol_binary = getBinaryTheta(interpNanImages(pol), tolerance, minNPix, mask);
        %         case 3
        %             pol_binary = logical(getBinaryTheta(interpNanImages(-pol), tolerance, minNPix, mask) ...
        %                 + getBinaryTheta(interpNanImages(pol), tolerance, minNPix, mask));
        %     end
        % 
        %     score_b(sid,itype) = bfscore(pol_binary, pol_binary_avg(:,:,itype));
        %     score_j(sid,itype) = jaccard(pol_binary, pol_binary_avg(:,:,itype));
        %     score_d(sid,itype) = dice(pol_binary, pol_binary_avg(:,:,itype));
        %     score_c(sid,itype) = chamferDistance(pol_binary, pol_binary_avg(:,:,itype));
        % 
        %     stats = regionprops(pol_binary, 'ConvexHull');
        %     pol_poly = polyshape(stats(1).ConvexHull);
        %     % pol_poly=polyshape([stats(1).ConvexHull; stats(2).ConvexHull]);
        %     % %tuningdist cannot compute > 1 boundary
        %     score_t(sid,itype) = turningdist(pol_poly, pol_poly_avg(:,:,itype));
        % end
    catch err
    end
end

subplot(131);
histogram(corr_pol,0.2:.05:1); hold on; histogram(corr_ecc,0.2:.05:1);
vline(corr_pol(10));
legend('polar','eccentricity');
xlabel('correlation to gnd avg'); ylabel('# subjects');

subplot(132);
histogram(overlap);
xlabel('Overlap V1 - V2+V3 [%]');

subplot(133);
histogram(coverage_v1);
xlabel('Coverage V1 [deg^2]');


selected_index = find((coverage_v1 > 80).*(overlap>80));
selected_id = subject_id(selected_index);


% tid = find(strcmp(subject_id, '214019'));
% disp(['Subject: ' subject_id{tid}]);
% disp(['Overlap: ' num2str(overlap(tid)) '[%]']);
% disp(['V1 Coverage: ' num2str(coverage_v1(tid)) '[deg^2]']); %[deg^2]
% disp(['corr_pol: ' num2str(corr_pol(tid))]);


save('subjectSelection','selected_id','selected_index','corr_ecc','corr_pol',...
    "coverage_v1",'coverage_v2v3','overlap','score_b','score_c','score_d','score_j','score_t');

ii=2;
subplot(511);
histogram(score_b(:,ii));hold on;histogram(score_b(selected_index,ii));vline(score_b(10));
subplot(512);
histogram(score_c(:,ii));hold on;histogram(score_c(selected_index,ii));vline(score_c(10));
subplot(513);
histogram(score_d(:,ii));hold on;histogram(score_d(selected_index,ii));vline(score_d(10));
subplot(514);
histogram(score_j(:,ii));hold on;histogram(score_j(selected_index,ii));vline(score_j(10));
subplot(515);
histogram(score_t(:,ii));hold on;histogram(score_t(selected_index,ii));vline(score_t(10));
