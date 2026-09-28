function [spCov, sphdom, actualCoverage] = getVisFieldCoverage(im,kmap_hor,kmap_vert,pixpermm)

%created from fusePatchesX.m
%Fuse patches if they are adjacent, the same sign, and unique regions of visual space

% OLapTh = 0.1;%0.15;
smoothInSpace = false;

xsize = size(kmap_hor,2)/pixpermm;  %Size of ROI mm
ysize = size(kmap_hor,1)/pixpermm;
xdum = linspace(0,xsize,size(kmap_hor,2)); ydum = linspace(0,ysize,size(kmap_hor,1));
[xdom ydom] = meshgrid(xdum,ydum); %two-dimensional domain

%%%First make a set of matrices that are not interpolated
[dhdx dhdy] = gradient(kmap_hor);
[dvdx dvdy] = gradient(kmap_vert);
% Jac = (dhdx.*dvdy - dvdx.*dhdy)*(pixpermm)^2;
graddir_hor = atan2(dhdy,dhdx);
graddir_vert = atan2(dvdy,dvdx);
vdiff = exp(1i*graddir_hor) .* exp(-1i*graddir_vert);
% Sereno = sin(angle(vdiff));
% imlab = bwlabel(im,4);  %Better to keep this as default (i.e. don' put in a 4)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if smoothInSpace
    hh = fspecial('gaussian',size(kmap_hor),2);
    kmap_horS = ifft2(fft2(kmap_hor).*abs(fft2(hh)));
    kmap_vertS = ifft2(fft2(kmap_vert).*abs(fft2(hh)));
else
    kmap_vertS = kmap_vert;
    kmap_horS = kmap_hor;
end

%%%Make Interpolated data to construct the visual space representations%%%
dim = size(kmap_horS);
U = 3; %upsampling factor
pixSize = 0.2;
xdum = linspace(xdom(1,1),xdom(1,end),U*dim(2)); ydum = linspace(ydom(1,1),ydom(end,1),U*dim(1));
[xdomI ydomI] = meshgrid(xdum,ydum); %upsample the domain
sphdom = -20:pixSize:20;%-90:90;  %create the domain for the sphere
kmap_hor_interp = interp2(xdom,ydom,kmap_horS,xdomI,ydomI,'spline');
kmap_vert_interp = interp2(xdom,ydom,kmap_vertS,xdomI,ydomI,'spline');
kmap_horI_idx = discretize(kmap_hor_interp, sphdom);
kmap_horI = sphdom(kmap_horI_idx);
kmap_vertI_idx = discretize(kmap_vert_interp, sphdom); %replaced round
kmap_vertI = sphdom(kmap_vertI_idx);

[dhdx dhdy] = gradient(kmap_hor_interp);
[dvdx dvdy] = gradient(kmap_vert_interp);
JacI = (dhdx.*dvdy - dvdx.*dhdy)*(pixpermm*U)^2;

imI = round(interp2(xdom,ydom,im,xdomI,ydomI,'nearest')); %interpolate to be the same size as the maps
imI(find(isnan(imI))) = 0;
% imlabI = bwlabel(imI,4); %Better to keep this as default (i.e. don't put in a 4)
% labdom = unique(imlabI);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[sphX sphY] = meshgrid(sphdom,sphdom);

%Create matrix of v. space coverage for each patch
SE = strel('disk',1,0);
% for i = 1:length(labdom)-1
patch = zeros(size(kmap_horI));
% id = find(imlabI == i);
% patch(id) = 1;
patch(imI==1) = 1;
patch = imdilate(patch,SE);
[spCov, actualCoverage] = overRep(kmap_horI,kmap_vertI,U,JacI,patch,sphdom,sphX,pixpermm);

% %Sereno map is messed up when ret is interpolated, so use the uninterped
% patch = zeros(size(kmap_hor));
% id = find(imlab == i);
% patch(id) = 1;
% AreaSign(i) = sign(mean(Sereno(id)));
% end
end


function [spCov JacCoverage ActualCoverage MagFac] = overRep(kmap_hor,kmap_vert,U,...
    Jac,patch,sphdom,sphX,pixpermm)

pixpermm = pixpermm*U;

N = length(sphdom);
pixSize = median(diff(sphdom));

posneg = sign(mean(Jac(find(patch))));
id = find(sign(Jac)~=posneg | Jac == 0);
Jac(id) = 0;
patch(id) = 0;
    
idpatch = find(patch);
JacCoverage = abs(sum(abs(Jac(idpatch))))/pixpermm^2; %deg^2 

% sphlocX = (kmap_hor(idpatch));
% sphlocX = sphlocX-sphdom(1)+1;
% sphlocY = (kmap_vert(idpatch));
% sphlocY = sphlocY-sphdom(1)+1;
% sphlocVec = N*(sphlocX-1) + sphlocY;
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

end