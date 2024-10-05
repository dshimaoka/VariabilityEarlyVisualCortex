function [im, vfs_th, VFS, fig] = getHumanAreasX(kmap_hor, kmap_vert,  smoothingFac, threshold, mask)

%% INPUTS
%kmap_hor - Map of horizontal retinotopic location %[deg], surrounded by nans
%kmap_vert - Map of vertical retinotopic location %[deg], surrounded by nans
%pixpermm = mm/pix of the retinotopy images
% The images in Garrett et al '14 were collected at 39 pixels/mm.  It is
% recommended that kmap_hor and kmap_vert be down/upsampled to this value
% before running.

%% OUTPUTS
%im is a binary image of the final set of patches

%TODO:
% imreconstruct instead of imopen
%cf. https://au.mathworks.com/help/images/marker-controlled-watershed-segmentation.html


if nargin < 5
    mask = ~isnan(kmap_hor);
end

kmap_hor(~mask) = 0;
kmap_vert(~mask) = 0;

if nargin < 4
    threshold = 1.5;
end

if nargin < 3
    smoothingFac = 3;
end

structuringElementRadius = 1;
structuringElementRadius2 = 1;
doImopen = false; %step 6
doThinning = true; %step 7
makeBoundary = false;

%% Compute visual field sign map

pixpermm = 1;
mmperpix = 1/pixpermm;

[dhdx dhdy] = gradient(kmap_hor);
[dvdx dvdy] = gradient(kmap_vert);

graddir_hor = atan2(dhdy,dhdx);
graddir_vert = atan2(dvdy,dvdx);

vdiff = exp(1i*graddir_hor) .* exp(-1i*graddir_vert); %Should be vert-hor, but the gradient in Matlab for y is opposite.
VFS = sin(angle(vdiff)); %Visual field sign map
id = find(isnan(VFS));
VFS(id) = 0;

%% smoothing VFS using opening-by-reconstruction
I = VFS;
SE = strel('disk',structuringElementRadius);

%opening-by-reconstruction	
Ie = imerode(I,SE);
Iobr = imreconstruct(Ie,I);

%Opening-Closing by Reconstruction
Iobrd = imdilate(Iobr,SE);
Iobrcbr = imreconstruct(imcomplement(Iobrd),imcomplement(Iobr));
Iobrcbr = imcomplement(Iobrcbr);
imagesc(Iobrcbr)


hh = fspecial('gaussian',size(VFS),smoothingFac); 
hh = hh/sum(hh(:));
%VFS = ifft2(fft2(VFS).*abs(fft2(hh)));  %Important to smooth before thresholding below
VFS = ifft2(fft2(Iobrcbr).*abs(fft2(hh)));


%% Plot retinotopic maps

xdom = (0:size(kmap_hor,2)-1)*mmperpix;
ydom = (0:size(kmap_hor,1)-1)*mmperpix;

fig=figure(10); clf
set(fig,'position',[0 0 1980 1080]);
ax(1)=subplot(3,4,1);
imagesc(xdom,ydom,kmap_hor,[-10 10]),
axis xy, colorbar
title('1. Horizontal (azim deg)')

ax(2)=subplot(3,4,2);
imagesc(xdom,ydom,kmap_vert,[-10 10]),
axis xy, colorbar
title('2. Vertical (alt deg)')

%% Plotting visual field sign and its threshold

figure(10), ax(3)=subplot(3,4,3);
imagesc(xdom,ydom,VFS), axis xy
colorbar
title(['3. VFS after filtering: ' num2str(smoothingFac)])

gradmag = abs(VFS);
figure(10), ax(4)=subplot(3,4,4); 

%threshSeg = threshold*nanstd(VFS(:));
%imseg = (sign(gradmag-threshSeg/2) + 1)/2;  %threshold visual field sign map at +/-1.5sig
[~, im_final] = getVisualBorder_ds(VFS,threshold,structuringElementRadius,structuringElementRadius2);
imseg = abs(im_final);
id = find(imseg);
imdum = imseg.*VFS; imdum(id) = imdum(id)+1.1;

ploteccmap(imdum,[.1 2.1],pixpermm);
colorbar off
axis xy
title(['4. +/' num2str(threshold) 'x s.d. ']) 


patchSign = getPatchSign(imseg,VFS);

figure(10), ax(5)=subplot(3,4,5),
ploteccmap(patchSign,[1.1 2.1],pixpermm);
title('watershed')
colorbar off
title('5. Threshold patches')

id = find(patchSign ~= 0);
patchSign(id) = sign(patchSign(id) - 1);


if doImopen
    SE = strel('disk',2,0);
    imseg = imopen(imseg,SE);

    patchSign = getPatchSign(imseg,VFS);

    figure(10), ax(6)=subplot(3,4,6),
    ploteccmap(patchSign,[1.1 2.1],pixpermm);
    title('watershed')
    colorbar off
    title('6. "Open" & set boundary')
end


%% Make boundary of visual cortex
if makeBoundary

%First pad the image with zeros because the "imclose" function does this
%wierd thing where it tries to "bleed" to the edge if the patch near it

Npad = 30;  %Arbitrary padding value.  May need more depending on image size and resolution
dim = size(imseg);
imsegpad = [zeros(dim(1),Npad) imseg zeros(dim(1),Npad)];
dim = size(imsegpad);
imsegpad = [zeros(Npad,dim(2)); imsegpad; zeros(Npad,dim(2))];

SE = strel('disk',10,0);
imbound = imclose(imsegpad,SE);

imbound = imfill(imbound); %often unnecessary, but sometimes there are small gaps need filling

% SE = strel('disk',5,0);
% imbound = imopen(imbound,SE);

SE = strel('disk',3,0);
imbound = imdilate(imbound,SE); %Dilate to account for original thresholding.
imbound = imfill(imbound);

%Remove the padding
imbound = imbound(Npad+1:end-Npad,Npad+1:end-Npad);
imbound(:,1) = 0; imbound(:,end) = 0; imbound(1,:) = 0;  imbound(end,:) = 0; 

%Only keep the "main" group of patches. Preveiously used opening (see above), but this is more robust:
bwlab = bwlabel(imbound,4);
labid = unique(bwlab);
for i = 1:length(labid)
   id = find(bwlab == labid(i));
   S(i) = length(id);
end
S(1) = 0; %To ignore the "surround patch"
[dum id] = max(S);
id = find(bwlab == labid(id));
imbound = 0*imbound;
imbound(id) = 1;

imbound = imbound .* mask;

else
    imbound = imdilate(mask, strel('disk',1));
end

imseg = imseg.*imbound;

%This is important in case a patch reaches the edge... we want it to be
%smaller than imbound
imseg(:,1:2) = 0; imseg(:,end-1:end) = 0; imseg(1:2,:) = 0;  imseg(end-1:end,:) = 0; 


if doImopen
    figure(10), ax(6)=subplot(3,4,6)
    hold on
    contour(xdom,ydom,imbound,[.5 .5],'k')
end

%% Morphological thinning to create borders that are one pixel wide
if doThinning
bordr = getThinning(imbound, imseg);

%Turn border map into patches
im = bwlabel(1-bordr,4);
im(find(im == 1)) = 0;
im = sign(im);

% bwlab = bwlabel(im,4);
% labid = unique(bwlab);
% for i = 1:length(labid)
%    id = find(bwlab == labid(i));
%    if length(id) < 30
%        im(id) = 0;
%    end
% end


patchSign = getPatchSign(im,VFS);

figure(10), ax(7)=subplot(3,4,7),
ploteccmap(patchSign,[1.1 2.1],pixpermm);
hold on, 
contour(xdom,ydom,im,[.5 .5],'k')
title('7. Thinning')
colorbar off
else 
    im = imseg;
    patchSign = getPatchSign(im,VFS);
end

%% Plot eccentricity map, with [0 0] defined as V1's center-of-mass

%AreaInfo.kmap_rad = kmap_rad;
AreaInfo.kmap_rad = atan(  sqrt( tan(kmap_hor*pi/180).^2 + (tan(kmap_vert*pi/180).^2)./(cos(kmap_hor*pi/180).^2)  )  )*180/pi;  %Eccentricity

ax(8)=subplot(3,4,8)
ploteccmap(AreaInfo.kmap_rad.*im,[0 5],pixpermm);
hold on
contour(xdom,ydom,im,[.5 .5],'k')
axis xy
title('8. Eccentricity map')


%% ID redundant patches and split them (criterion #2)

im = splitPatchesX(im,kmap_hor,kmap_vert,AreaInfo.kmap_rad,pixpermm); 

    % %Remake the border with thinning
    % bordr = getThinning(imbound, im);
    % 
    % %Turn border map into patches
    % im = bwlabel(1-bordr,4);
    % im(find(im == 1)) = 0;
    % im = sign(im);

if doImopen
    SE = strel('disk',2);
    im = imopen(im,SE);
end

%% ID adjacent patches of the same VFS and fuse them if not redundant (criterion #3)

[im fuseflag] = fusePatchesX(im,kmap_hor,kmap_vert,pixpermm); 

figure(10), ax(9)=subplot(3,4,9),
ploteccmap(im.*AreaInfo.kmap_rad,[0 5],pixpermm);
title('9. Split redundant patches. Fuse exclusive patches.')


%% trim small patches
imlab = bwlabel(im,4);
for q=1:numel(unique(imlab))
    if sum(im(imlab==q)) < 10 %#pixels
        im(imlab==q) = 0;
    end
end


vfs_th = getPatchSign(im,VFS);
vfs_th(vfs_th == 2.1) = 1;
vfs_th(vfs_th == 0.1) = -1;

%%

subplot(3,4,1)
hold on,
contour(xdom,ydom,im,[.5 .5],'k')

subplot(3,4,2)
hold on,
contour(xdom,ydom,im,[.5 .5],'k')

subplot(3,4,3)
hold on,
contour(xdom,ydom,im,[.5 .5],'k')

[patchSign areaSign] = getPatchSign(im,VFS);
figure(10), ax(10)=subplot(3,4,10)
ploteccmap(patchSign,[1.1 2.1],pixpermm); colorbar off
hold on
contour(xdom,ydom,im,[.5 .5],'k')
axis xy

title('10. visual areas')


%% Plot contours

figure(10)
ax(11)=subplot(3,4,11)
contour(xdom,ydom,kmap_vert.*im,[-90:4:90],'r')
hold on
contour(xdom,ydom,kmap_hor.*im,[-90:4:90],'k')
axis ij
title('Red: Vertical Ret;  Black: Horizontal Ret')
axis xy
xlim([xdom(1) xdom(end)]), ylim([ydom(1) ydom(end)])

%% Get magnification factor images
[JacIm prefAxisMF Distort] = getMagFactors(kmap_hor,kmap_vert,pixpermm);

figure(10)
ax(12)=subplot(3,4,12)
plotmap(im.*sqrt(1./abs(JacIm)),[sqrt(.000001) sqrt(.003)],pixpermm);  %This doesn't work
title('Mag fac (mm2/deg2)')

dim = size(prefAxisMF);
DdomX = 10:10:dim(2);
DdomY = 10:10:dim(1);
prefAxisMF = prefAxisMF(DdomY,DdomX);
Distort = Distort(DdomY,DdomX);

figure(10)
subplot(3,4,12)
hold on,
contour(xdom,ydom,im,[.5 .5],'k');
for i = 1:length(DdomX)
    for j = 1:length(DdomY)
        
        xpart = 5*Distort(j,i)*cos(prefAxisMF(j,i)*pi/180);
        ypart = 5*Distort(j,i)*sin(prefAxisMF(j,i)*pi/180);
        
        if im(DdomY(j),DdomX(i))
            hold on, plot([DdomX(i)-xpart DdomX(i)+xpart]*mmperpix,[DdomY(j)-ypart DdomY(j)+ypart]*mmperpix,'k')
        end
        
    end
end
linkaxes(ax(:));


function imout = ploteccmap(im,rng,pixpermm)

%This assumes that the zeros are the background
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


image(xdom,ydom,imout), axis xy

eccdom = round(linspace(rng(1),rng(2),5));
for i = 1:length(eccdom)
    domcell{i} = eccdom(i);
end
iddom = linspace(1,64,length(eccdom));
colorbar('YTick',iddom,'YTickLabel',domcell)

function imout = plotehorvertmap(im,rng)

%This assumes that the zeros are the background

xdom = (0:size(im,2)-1)*mmperpix;
ydom = (0:size(im,1)-1)*mmperpix;

bg = ones(size(im));
bgid = find(im == 0);
bg(bgid) = 0;

im(find(im>rng(2))) = rng(2);
im(find(im<rng(1))) = rng(1);
im = im-rng(1);
im = im/(rng(2)-rng(1));

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


image(xdom,ydom,imout), axis xy

eccdom = round(linspace(rng(1),rng(2),5));
for i = 1:length(eccdom)
    domcell{i} = eccdom(i);
end
iddom = linspace(1,64,length(eccdom));
colorbar('YTick',iddom,'YTickLabel',domcell)

hold on,
contour(xdom,ydom,bg,[.5 .5],'k')



function [patchSign areaSign] = getPatchSign(im,imsign)

imlabel = bwlabel(im,4);
areaID = unique(imlabel);
patchSign = zeros(size(imlabel));
for i = 2:length(areaID)
   id = find(imlabel == areaID(i));
   m = mean(imsign(id));
   areaSign(i-1) = sign(m);
   patchSign(id) = sign(m)+1.1;
end



function bordr = getThinning(imbound, imseg)
%bordr = imbound-imseg;
bordr = imbound - imerode(imseg<0, strel('disk',1)) - imerode(imseg>0, strel('disk',1)); %more robust
bordr = bwmorph(bordr,'thin',Inf);
bordr = bwmorph(bordr,'spur',4);

bordr = bwmorph(bordr,'thin',Inf);

