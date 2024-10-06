function im = splitPatchesX(im,kmap_hor,kmap_vert,kmap_rad,pixpermm)

debug = false;
CovOverlapTh = 1.1;%as described in Garrett 2014
smoothInSpace = false;
doImopen = false;

xsize = size(kmap_hor,2)/pixpermm;  %Size of ROI mm
ysize = size(kmap_hor,1)/pixpermm; 
xdum = linspace(0,xsize,size(kmap_hor,2)); ydum = linspace(0,ysize,size(kmap_hor,1)); 
[xdom ydom] = meshgrid(xdum,ydum); %two-dimensional domain

if smoothInSpace
    kmap_rad = smoothPatchesX(kmap_rad,im); %smooth the larger patches

    hh = fspecial('gaussian',size(kmap_hor),2);
    kmap_horS = ifft2(fft2(kmap_hor).*abs(fft2(hh)));
    kmap_vertS = ifft2(fft2(kmap_vert).*abs(fft2(hh)));
else
    kmap_rad = kmap_rad.*im;
    kmap_rad(im==0) = 45;
    kmap_horS = kmap_hor;
    kmap_vertS = kmap_vert;
end

[dhdx dhdy] = gradient(kmap_horS);
[dvdx dvdy] = gradient(kmap_vertS);
Jac = (dhdx.*dvdy - dvdx.*dhdy)*pixpermm^2; %deg^2/mm^2  %magnification factor is determinant of Jacobian

%%%Make Interpolated data to construct the visual space representations%%%
dim = size(kmap_horS);
U = 3; %upsample factor
pixSize = .2; %pixel size in [deg] in visual field
xdum = linspace(xdom(1,1),xdom(1,end),U*dim(2)); ydum = linspace(ydom(1,1),ydom(end,1),U*dim(1));
[xdomI ydomI] = meshgrid(xdum,ydum); %upsample the domain
sphdom = -20:pixSize:20;%-90:90;  %create the domain for the sphere
kmap_hor_interp = interp2(xdom,ydom,kmap_horS,xdomI,ydomI,'spline');
kmap_vert_interp = interp2(xdom,ydom,kmap_vertS,xdomI,ydomI,'spline');
kmap_horI_idx = discretize(kmap_hor_interp, sphdom);
kmap_horI = sphdom(kmap_horI_idx);
kmap_vertI_idx = discretize(kmap_vert_interp, sphdom); %replaced round
kmap_vertI = sphdom(kmap_vertI_idx);
kmap_radI = (interp2(xdom,ydom,kmap_rad,xdomI,ydomI,'spline'));%removed round

[dhdx dhdy] = gradient(kmap_hor_interp);
[dvdx dvdy] = gradient(kmap_vert_interp);
JacI = (dhdx.*dvdy - dvdx.*dhdy)*(pixpermm*U)^2;
% [xdomI ydomI] = meshgrid(xdum,ydum); %upsample the domain
% kmap_horI = (interp2(xdom,ydom,kmap_horS,xdomI,ydomI,'spline')); %removed round
% kmap_vertI =(interp2(xdom,ydom,kmap_vertS,xdomI,ydomI,'spline'));%removed round
% 
% [dhdx dhdy] = gradient(kmap_horI);
% [dvdx dvdy] = gradient(kmap_vertI);
% JacI = (dhdx.*dvdy - dvdx.*dhdy)*(pixpermm*U)^2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if doImopen
    SE = strel('disk',1,0);
    im = imopen(sign(im),SE);
    imlab = bwlabel(im,4);
    imdom = unique(imlab);

    SE = strel('disk',1,0);
    im = imerode(sign(im),SE);
end

[sphX sphY] = meshgrid(sphdom,sphdom);

%% Find patches to 'split'
clear spCov

R = 15;%30; %Find local min within central R deg

imI = round(interp2(xdom,ydom,im,xdomI,ydomI,'nearest')); %interpolate to be the same size as the maps

%%%%%Now proceed with splitting based on no. of minima
imlab = bwlabel(im,4); 
imdom = unique(imlab);

imlabI = bwlabel(imI,4); 

centerPatch = getCenterPatch(kmap_rad,im,R);
centerPatchI = getCenterPatch(kmap_radI,imI,R);

for q = 1:length(imdom)-1 %loop through each patch ("visual area")
    
    idpatch = find(imlab == q & centerPatch);        
    dumpatch = zeros(size(im));
    dumpatch(idpatch) = 1;
    
    idpatchI = find(imlabI == q & centerPatchI);   
    dumpatchI = zeros(size(imI));
    dumpatchI(idpatchI) = 1;

    
    %figure,imagesc(dumpatch)
       % figure(2);
       %  nexttile;
       %  imagesc(dumpatchI); title(q);

    Nmin = 1;
    if ~isempty(find(idpatch))
        
        
        %%Determine if it has a overlapping representation of visual space%%%%%%

        [spCov JacCoverage ActualCoverage MagFac] = overRep(kmap_horI,kmap_vertI,U,JacI,dumpatchI,sphdom,sphX,pixpermm);
        CovOverlap = JacCoverage/ActualCoverage;

     
        
        if CovOverlap > CovOverlapTh%.999
            
            %figure, imagesc(dumpatch)
            figure
            subplot(2,2,1),
            imagesc(dumpatch.*kmap_hor); colorbar;
            title('azimuth of redundant patch on brain');

            subplot(2,2,2),
            imagesc(dumpatch.*kmap_vert); colorbar;
            title('altitude');
            
           subplot(2,2,3);
           domX = sphX*pi/180; %azimuth
           domY = sphY*pi/180; %altitude
            [c h] = contour(domX(1,:)*180/pi,(domY(:,1))*180/pi,spCov,[.5 .5],'LineColor',[1 0 0]);
           title('coverage in visual field');      

            hor_cent = 0; %median(kmap_hor(find(dumpatch)));
            vert_cent = 0; %median(kmap_vert(find(dumpatch)));
            
            kmap_rad_cent = sqrt((kmap_hor-hor_cent).^2 + (kmap_vert-vert_cent).^2);
            
            kmap_rad_dum = zeros(size(kmap_rad));
            kmap_rad_dum(idpatch) = kmap_rad_cent(idpatch);

            [Nmin minpatch centerPatch2 Rdiscrete] = getNlocalmin(idpatch,R,kmap_rad_dum);

             %executes splitting according to centerPatch2
            % [im splitflag Nsplit] = resetPatch(im,centerPatch2,imlab,q);
            im = im - dumpatch + centerPatch2;
          
        end
    end


    if Nmin > 1
        id = find(imlab == q);
        dumpatch = zeros(size(im)); dumpatch(id) = 1;

        figure,
        subplot(1,3,1), ploteccmap(dumpatch.*kmap_rad_cent,[0 R],1,pixpermm);
        title('Smoothed eccentricity map'), colorbar off
        ylabel('mm'), xlabel('mm')

        id = find(Rdiscrete == median(Rdiscrete(:))); Rdiscrete(id) = 0;
        subplot(1,3,2), ploteccmap(dumpatch.*Rdiscrete,[0 R],1,pixpermm);
        hold on, contour(xdom,ydom,minpatch,[.5 .5],'k')
        title(['Discretized map; ' num2str(Nmin) ' minima found']), colorbar off

        subplot(1,3,3), ploteccmap(dumpatch.*kmap_rad_cent.*im,[0 R],1,pixpermm);
        title('Flood the patch with watershed')
    end
end



%% Compute level of over-representation after splitting

imlab = bwlabel(im,4); 
imdom = unique(imlab);

imI = round(interp2(xdom,ydom,im,xdomI,ydomI,'nearest')); %interpolate to be the same size as the maps
imI(find(isnan(imI))) = 0;
imlabI = bwlabel(imI,4);

%R = 35;

centerPatch = getCenterPatch(kmap_rad,im,R);
centerPatchI = getCenterPatch(kmap_radI,imI,R);
clear spCov JacCoverage ActualCoverage MagFac

for q = 1:length(imdom)-1 %loop through each patch ("visual area")

    idpatch = find(imlab == q & centerPatch);
    dumpatch = zeros(size(im));
    dumpatch(idpatch) = 1;
    
    idpatchI = find(imlabI == q & centerPatchI);   
    dumpatchI = zeros(size(imI));
    dumpatchI(idpatchI) = 1;
    
    %figure,imagesc(dumpatchI)

    [spCov{q} JacCoverage(q) ActualCoverage(q) MagFac(q)] = overRep(kmap_horI,kmap_vertI,U,JacI,dumpatchI,sphdom,sphX,pixpermm);

    CovOverlap = JacCoverage(q)/ActualCoverage(q);

end

if debug
    figure,
    scatter(JacCoverage,ActualCoverage)
    hold on
    plot([0 max(JacCoverage)], [0 max(JacCoverage)],'k')
    xlabel('Jacobian integral (deg^2)')
    ylabel('Actual Coverage (deg^2)')
end

function [spCov JacCoverage ActualCoverage MagFac] = overRep(kmap_hor,kmap_vert,U,...
    Jac,patch,sphdom,sphX,pixpermm)
% ActualCoverage becomes larger when pixSize<1 but JacCoverage is constant

pixpermm = pixpermm*U;

N = length(sphdom);
pixSize = median(diff(sphdom));

posneg = sign(mean(Jac(find(patch))));
id = find(sign(Jac)~=posneg | Jac == 0);
Jac(id) = 0;
patch(id) = 0;
    
idpatch = find(patch);
JacCoverage = abs(sum(abs(Jac(idpatch))))/pixpermm^2; %deg^2

sphlocX = zeros(numel(idpatch),1);
sphlocY = zeros(numel(idpatch),1);
for pp = 1:numel(idpatch)
    [~,sphlocX(pp)] = intersect(unique(sphX), kmap_hor(idpatch(pp)));
    [~,sphlocY(pp)] = intersect(unique(sphX), kmap_vert(idpatch(pp)));
end
sphlocVec = sub2ind(size(sphX),sphlocX, sphlocY);

spCov = zeros(size(sphX)); %a matrix that represents the visual field
spCov(sphlocVec) = 1;
spCov = imfill(spCov);
SE = strel('disk', round(1/pixSize),0);
spCov = imclose(spCov,SE);
spCov = imfill(spCov);
%spCov = medfilt2(spCov,[3 3]);
ActualCoverage = sum(spCov(:)).*(pixSize^2); %deg^2
MagFac = ActualCoverage/length(idpatch);


function centerPatch = getCenterPatch(kmap_rad,im,R)
smoothInSpace = false;

id = find(kmap_rad<R);  %Find pixels near the center of visual space
centerPatch = zeros(size(im));
centerPatch(id) = 1;  %Make a mask for them
centerPatch = centerPatch.*im;  

if smoothInSpace
    SE = strel('disk',2,0);
    centerPatch = imopen(centerPatch,SE); %clean it up
    centerPatch = medfilt2(centerPatch,[3 3]);
end

function [Nmin minpatch newpatches rad] = getNlocalmin(idpatch,Rmax,kmap_rad)

%Determine number of local minima

dum = zeros(size(kmap_rad));
dum(idpatch) = 1;
idnopatch = find(dum == 0);

%discretize radius map into 10, making local minima detection easier
kr = kmap_rad(idpatch);
threshdom = min(kr)-1;
for prc = 2:10:90
    threshdom = [threshdom prctile(kr,prc)];
end
threshdom = [threshdom max(kr)+1]; 

for i = 1:length(threshdom)-1
   id = find(kmap_rad>threshdom(i) & kmap_rad<threshdom(i+1));
   kmap_rad(id) = mean(kmap_rad(id));
end

kmap_rad(idnopatch) = max(kmap_rad(idpatch));
SE = strel('disk',3,0);
kmap_rad = imopen(kmap_rad,SE);

%kmap_rad = medfilt2(kmap_rad,[3 3]);

rad = zeros(size(kmap_rad));
rad(idnopatch) = Rmax;
rad(idpatch) = kmap_rad(idpatch);


%medR = ceil(length(idpatch)/400);
medR = 3;
rad = medfilt2(rad,[medR medR]);  %Really important to do this after applying Rmax boundary. It gets rid of the tiny local minima on the edges
%rad = medfilt2(rad,[3 3]);

dumpatch = zeros(size(kmap_rad));
dumpatch(idpatch) = 1;

minpatch = imregionalmin(rad,8);
minpatch = minpatch.*dumpatch;

D = round(sqrt(length(idpatch))/20);
%D = 1;
SE = strel('disk',D,0);
minpatch = imdilate(minpatch,SE);
minpatch = minpatch.*dumpatch;

%figure,imagesc(dumpatch.*kmap_rad)

imlabel = bwlabel(minpatch,4);
idlabel = unique(imlabel);
Nmin = length(idlabel)-1;

imlabel = bwlabel(minpatch,4);
idlabel = unique(imlabel);
Nmin = length(idlabel)-1;
    
SE = strel('disk',3,0);
dumpatch2 = imdilate(dumpatch,SE);

rad2 = imimposemin(rad, minpatch);
id = find(1-dumpatch);
rad2(id) = Rmax;  %reset, in case min is on the edge

id = find(~dumpatch2);
rad2(id) = -inf;

newpatches = watershed(rad2);

id = find(newpatches == 1); %change 'im' to a binary set of patches
newpatches(id) = 0;
id = find(newpatches > 0);
newpatches(id) = 1;

newpatches = double(newpatches) .* dum; %same shape as the input 



function imout = ploteccmap(im,rng,DS,pixpermm)
%rng: range of colormap
%DS: down-sampling factor
%This assumes that the zeros are the background

im = im(DS:DS:end,DS:DS:end);

mmperpix = 1/pixpermm;
xdom = (0:size(im,2)-1)*mmperpix;
ydom = (0:size(im,1)-1)*mmperpix;

bg = ones(size(im));
bgid = find(im == 0);
bg(bgid) = 0;

im(find(im>rng(2))) = rng(2);
im = im/rng(2);

im = round(im*63+1);

im(bgid) = NaN;

dim = size(im);
jetid = jet;
imout = zeros(dim(1),dim(2),3);
for i = 1:dim(1)
    for j = 1:dim(2)
        
        if isnan(im(i,j))
            imout(i,j,:) = [1 1 1];
        else
            imout(i,j,:) = jetid(im(i,j),:);
        end
    end
end


image(xdom,ydom,imout), axis image

eccdom = round(linspace(rng(1),rng(2),5));
for i = 1:length(eccdom)
    domcell{i} = eccdom(i);
end
iddom = linspace(1,64,length(eccdom));
colorbar('YTick',iddom,'YTickLabel',domcell)

hold on,
contour(xdom,ydom,bg,[.5 .5],'k')



