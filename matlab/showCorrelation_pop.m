server = '/mnt/dshi0006_market/VariabilityEarlyVisualCortex';
subject_id = getSubjectId;
parValues = [10 20 40 80 160];

error_id = {};
corr_v_all = nan(numel(parValues),numel(parValues),numel(subject_id),2);
corr_s_all = nan(numel(parValues),numel(parValues),numel(subject_id),2);
for isub = 1:numel(subject_id)
    thisSubject = subject_id{isub};
    disp(thisSubject);
    try
        load(fullfile(server, thisSubject ,['arealBorder_' thisSubject '.mat']),...
            'areaMatrix','grid_altitude_i','grid_azimuth_i');
        for ib1 = 1:numel(parValues)
            b1 = parValues(ib1);

            for ib2 = 1:numel(parValues)
            b2 = parValues(ib2);

                load(fullfile(server, thisSubject, ['summary_V+D_' thisSubject '_b1_' num2str(b1) '_b2_' num2str(b2) '_vs.mat']), ...
                    'result2d_v','result2d_s');

                tgtPixIdx = union(find(areaMatrix{2}),find(areaMatrix{3}));
                %tgtPixIdx = find(areaMatrix{1});

                [ecc, pol] = cart2pol(grid_azimuth_i, grid_altitude_i);
                [ecc_v, pol_v] = cart2pol(result2d_v(:,:,1)', result2d_v(:,:,2)');
                [ecc_s, pol_s] = cart2pol(result2d_s(:,:,1)', result2d_s(:,:,2)');


                corr_ecc_v = corr(ecc(tgtPixIdx), ecc_v(tgtPixIdx),'type','Spearman');
                corr_ecc_s = corr(ecc(tgtPixIdx), ecc_s(tgtPixIdx),'type','Spearman');

                corr_pol_v = corr(pol(tgtPixIdx), pol_v(tgtPixIdx),'type','Spearman');
                corr_pol_s = corr(pol(tgtPixIdx), pol_s(tgtPixIdx),'type','Spearman');

                corr_v_all(ib1,ib2,isub,1) = corr_ecc_v;
                corr_s_all(ib1,ib2,isub,1) = corr_ecc_s;

                corr_v_all(ib1,ib2,isub,2) = corr_pol_v;
                corr_s_all(ib1,ib2,isub,2) = corr_pol_s;
            end
        end
    catch err
        disp(err);
        error_id{numel(error_id)+1} = thisSubject;
    end
end

subplot(121);
imagesc(squeeze(nanmean(corr_v_all(:,:,:,1) - corr_s_all(:,:,:,1),3)));
axis xy;
squareplot;
xlabel('b1');ylabel('b2');

subplot(122);
imagesc(squeeze(nanmean(corr_v_all(:,:,:,2) - corr_s_all(:,:,:,2),3)));
axis xy;
squareplot;
xlabel('b1');ylabel('b2');

[corr_pol_diff, subject_id_pol] = sort(corr_v_all(1,5,:,2)-corr_s_all(1,5,:,2));
