server = '/mnt/dshi0006_market/VariabilityEarlyVisualCortex';
subject = 'avg';
values = [10 20 40 80 160];

corr_all_v = zeros(numel(values));
corr_all_s = zeros(numel(values));

for ib1 = 1:numel(values)
    for ib2 = 1:numel(values)
        b1 = values(ib1);
        b2 = values(ib2);

        load(fullfile(server, subject ,['arealBorder_' subject '.mat']),...
            'areaMatrix','grid_altitude_i','grid_azimuth_i');
        load(fullfile(server, subject, ['summary_V+D_' subject '_b1_' num2str(b1) '_b2_' num2str(b2) '_vs.mat']), ...
            'result2d_v','result2d_s');

        tgtPixIdx = union(find(areaMatrix{2}),find(areaMatrix{3}));
        %tgtPixIdx = find(areaMatrix{1});

        [ecc, pol] = cart2pol(grid_azimuth_i, grid_altitude_i);
        [ecc_v, pol_v] = cart2pol(result2d_v(:,:,1)', result2d_v(:,:,2)');
        [ecc_s, pol_s] = cart2pol(result2d_s(:,:,1)', result2d_s(:,:,2)');



        corr_v = corr(pol(tgtPixIdx), pol_v(tgtPixIdx),'type','Spearman');
        corr_s = corr(pol(tgtPixIdx), pol_s(tgtPixIdx),'type','Spearman');

        corr_all_v(ib1,ib2)=corr_v;
        corr_all_s(ib1,ib2)=corr_s;
    end
end
subplot(131);
imagesc(corr_all_s);axis xy;
set(gca,'xtick',1:numel(values),'XTickLabel',values,'ytick',1:numel(values),'YTickLabel',values);
xlabel('b2');ylabel('b1');
clim([0.835 0.865]);mcolorbar;
title('Spearman correlation, surface minimization');

subplot(132);
imagesc(corr_all_v);axis xy;
set(gca,'xtick',1:numel(values),'XTickLabel',values,'ytick',1:numel(values),'YTickLabel',values);
xlabel('b2');ylabel('b1');
clim([0.835 0.865]);mcolorbar;
title('volume minimization');

subplot(133);
imagesc(corr_all_v-corr_all_s);axis xy;
set(gca,'xtick',1:numel(values),'XTickLabel',values,'ytick',1:numel(values),'YTickLabel',values);
xlabel('b2');ylabel('b1');
title('volume-surface');
clim([-6e-3 6e-3]);mcolorbar;
squareplots