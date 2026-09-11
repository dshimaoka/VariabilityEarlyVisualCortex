server = '/mnt/dshi0006_market/VariabilityEarlyVisualCortex';
subject = '910241';
b1 = 160;
b2 = 10;

load(fullfile(server, subject ,['arealBorder_' subject '.mat']),...
    'areaMatrix','grid_altitude_i','grid_azimuth_i');
load(fullfile(server, subject, ['summary_' subject 'V+D_' subject '_b1_' num2str(b1) '_b2_' num2str(b2) '.mat']), ...
    'result2d','result2d_flat');

tgtPixIdx = union(find(areaMatrix{2}),find(areaMatrix{3}));
%tgtPixIdx = find(areaMatrix{1});

[ecc, pol] = cart2pol(grid_azimuth_i, grid_altitude_i);
[ecc_v, pol_v] = cart2pol(result2d(:,:,1)', result2d(:,:,2)');
[ecc_s, pol_s] = cart2pol(result2d_flat(:,:,1)', result2d_flat(:,:,2)');


figure('position',[0 0 1400 700]);
ax = axes; 
ax(1)=subplot(231);
imagesc(pol);title('empirical');
ax(2)=subplot(232);
imagesc(pol_v);title('volume');
ax(3)=subplot(233);
imagesc(pol_s);title('surface');
linkcaxes(ax,[-90 90]);
mcolorbar(gca, .5);

subplot(212);
plot(pol(tgtPixIdx), pol_v(tgtPixIdx),'b.',pol(tgtPixIdx), pol_s(tgtPixIdx),'r.');
corr_v = corr(pol(tgtPixIdx), pol_v(tgtPixIdx),'type','Spearman');
corr_s = corr(pol(tgtPixIdx), pol_s(tgtPixIdx),'type','Spearman');

squareplot;
xlabel('empirical');ylabel('simulated');
title(['v: ' num2str(corr_v) ', s: ' num2str(corr_s)]);
legend('volume','surface','Location','northwest');

screen2png([subject '_b1_' num2str(b1) '_b2_' num2str(b2) '_2026March']);

[corr_pol_diff, subject_id_pol] = sort(corr_v_all(5,1,:,2)-corr_s_all(5,1,:,2));


