all_ids = {'114823'};%#,'157336','585256','581450','725751','avg']; #from Ribeiro 2023 Fig1
tgt = "V+D";
loadDir = '/mnt/dshi0006_market/VariabilityEarlyVisualCortex/';
numb1= 10;
numb2 = 10;
type = '';%'_flat';

id = 1;

b1 = 0.01*2.^(0:numb1-1);
b2 = 0.01*2.^(0:numb2-1);
     
%original data
%original = load(fullfile(loadDir, subject_id, sprintf('geometry_retinotopy_%s.mat', subject_id)));        
%
%areal border
original = load(fullfile(loadDir, subject_id, sprintf('arealBorder_%s.mat',subject_id)));
theta_ori = 180/pi*atan2(original.grid_altitude_i, original.grid_azimuth_i);

tolerance = 45;%deg
minNPix = 20; %number of pixels
mask = original.mask; %all areas including V1

theta_ori_binary = getBinaryTheta(theta_ori, tolerance, minNPix, mask);

stats = regionprops(theta_ori_binary, 'ConvexHull');
theta_ori_poly=polyshape(stats(1).ConvexHull); %only the largest shape

%%
costs_final = zeros(numb1,numb2,3);
result2d_polar_all = zeros(100,100, numb1,numb2);
result2d_binary_all = zeros(100,100, numb1, numb2);
subject_id = all_ids{id};
similarity = zeros(numb1, numb2);
for i1=1:numb1
    for i2=1:numb2
        suffix = sprintf('%s_%s_b1_%d_b2_%d',tgt,subject_id,1e3*b1(i1),1e3*b2(i2));
        
        loadName = sprintf('summary_%s%s.mat', subject_id, suffix);
        load(fullfile(loadDir, subject_id, loadName),'result','result_flat','result2d');

        
        if strcmp(type,'_flat')
            result = result_flat;
        end
        costs_final(i1,  i2, 1:3) = [result{2}(end) result{3}(end) result{4}(end)];

        theta = 180/pi*atan2(result2d(:,:,2), result2d(:,:,1))';
        
       
        %% convert to binary image
        theta_interp = interpNanImages(theta); %fill areal boundary
        theta_binary = getBinaryTheta(theta_interp, tolerance, minNPix, mask);

        %% compute similarity
        % score = bfscore(theta_binary, theta_ori_binary);
        % score = jaccard(theta_binary, theta_ori_binary);
        score = dice(theta_binary, theta_ori_binary);

        
        % stats = regionprops(theta_binary, 'ConvexHull');
        % theta_poly=polyshape(stats(1).ConvexHull); 
        % % theta_poly=polyshape([stats(1).ConvexHull; stats(2).ConvexHull]);
        % % %tuningdist cannot compute > 1 boundary
        % td = turningdist(theta_poly, theta_ori_poly);

        
        %% save result
         result2d_polar_all(:,:,i1,i2) = theta_interp;
         result2d_binary_all(:,:,i1,i2) = theta_binary;
         similarity(i1,i2) = score;

    end
end

subplot(311);
imagesc(log(-costs_final(:,:,1)));xlabel('b2');ylabel('b1');axis equal tight xy;colorbar
set(gca,'xtick',1:numb2,'ytick',1:numb1,'XTickLabel',b2,'yTickLabel',b1);
title('landmark(log)')

subplot(312);
imagesc(log(costs_final(:,:,2)));xlabel('b2');ylabel('b1');axis equal tight xy;colorbar
set(gca,'xtick',1:numb2,'ytick',1:numb1,'XTickLabel',b2,'yTickLabel',b1)
title('log(reg1)')

subplot(313);
imagesc(log(costs_final(:,:,3)));xlabel('b2');ylabel('b1');axis equal tight xy;colorbar
set(gca,'xtick',1:numb2,'ytick',1:numb1,'XTickLabel',b2,'yTickLabel',b1)
title('log(reg2)');

screen2png(['costFunctions_' subject_id type])


%% all polar plot 
figure('Position',[0 0 1920 960]);
for i1=1:numb1
    for i2=1:numb2
        ax(i1,i2)=subplot_tight(numb1,numb2, numb2*(10 - i1) + i2);
        %imagesc(squeeze(result2d_polar_all(:,:,i1,i2))');
        %clim([90 270]);
        imagesc(squeeze(result2d_binary_all(:,:,i1,i2)));
        axis equal tight xy off;

        %title([num2str(b1(i1)) ' ' num2str(b2(i2))]);
    end
end
mcolorbar;
linkaxes(ax);

screen2png(['theta_binary_' subject_id type])


%% similarity score
imagesc(similarity);xlabel('b2');ylabel('b1');axis equal tight xy;colorbar
set(gca,'xtick',1:numb2,'ytick',1:numb1,'XTickLabel',b2,'yTickLabel',b1);
title('turningdist');
title('similarity');

screen2png(['turninigdist_' subject_id type])
screen2png(['similarity_' subject_id type])
