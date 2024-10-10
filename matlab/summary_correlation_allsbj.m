subject_id = {'114823','157336','585256','581450','725751'};
saveDir = '/mnt/dshi0006_market/VariabilityEarlyVisualCortex/';

b1 = 0.01*2.^(0:4);
b2 = 0.01*2.^(0:4);

for sid= 1:numel(subject_id) 
    summary =  load(fullfile(saveDir, subject_id{sid}, ['summary_correlation_' subject_id{sid} '.mat']));

    corr_azimuth(:,:,sid) = summary.corr_azimuth;
    corr_altitude(:,:,sid) = summary.corr_altitude;
    corr_pa(:,:,sid)=summary.corr_pa;
    
    corr_azimuth_euc(:,:,sid) = summary.corr_azimuth_euc;
    corr_altitude_euc(:,:,sid) = summary.corr_altitude_euc;
    corr_pa_euc(:,:,sid)=summary.corr_pa_euc;

    corr_azimuth_flat(:,:,sid) = summary.corr_azimuth_flat;
    corr_altitude_flat(:,:,sid) = summary.corr_altitude_flat;
    corr_pa_flat(:,:,sid)=summary.corr_pa_flat;
    
end

subplot(3,3,1);
imagesc(mean(corr_azimuth,3)); axis equal tight xy; 
set(gca,'xticklabel',b2,'YTickLabel',b1)
clim([0.5 1]);
title('azimuth');
ylabel('min path length')

subplot(3,3,2);
imagesc(mean(corr_altitude,3)); axis equal tight xy; 
set(gca,'xticklabel',b2,'YTickLabel',b1)
clim([0.5 1]);
title('altitude')

subplot(3,3,3);
imagesc(mean(corr_pa,3)); axis equal tight xy; 
set(gca,'xticklabel',b2,'YTickLabel',b1)
clim([0.5 1]);
title('polar angle');

subplot(3,3,4);
imagesc(mean(corr_azimuth_flat,3)); axis equal tight xy; 
set(gca,'xticklabel',b2,'YTickLabel',b1)
clim([0.5 1]);
ylabel('flat surface')

subplot(3,3,5);
imagesc(mean(corr_altitude_flat,3)); axis equal tight xy; 
set(gca,'xticklabel',b2,'YTickLabel',b1)
clim([0.5 1]);

subplot(3,3,6);
imagesc(mean(corr_pa_flat,3)); axis equal tight xy; 
set(gca,'xticklabel',b2,'YTickLabel',b1)
clim([0.5 1]);

subplot(3,3,7);
imagesc(mean(corr_azimuth_euc,3)); axis equal tight xy; 
set(gca,'xticklabel',b2,'YTickLabel',b1)
clim([0.5 1]);
ylabel('euclidean dist');

subplot(3,3,8);
imagesc(mean(corr_altitude_euc,3)); axis equal tight xy; 
set(gca,'xticklabel',b2,'YTickLabel',b1)
clim([0.5 1]);

subplot(3,3,9);
imagesc(mean(corr_pa_euc,3)); axis equal tight xy; 
set(gca,'xticklabel',b2,'YTickLabel',b1)
clim([0.5 1]);
xlabel('b2'); ylabel('b1'); mcolorbar;


screen2png('summary_correlation_allsbj');