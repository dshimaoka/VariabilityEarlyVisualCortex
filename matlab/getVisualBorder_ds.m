function [signMapThreshold, im_final] = getVisualBorder_ds(signMap,...
    threshold,structuringElementRadius,structuringElementRadius2)
%[signMap, signMapThreshold, im_final] = getVisualBorder(signMap,...
%    threshold,structuringElementRadius,structuringElementRadius2)
% compute visual area order from VFS map
% 2018-12-11 created from getVisualAreas_DS.m
%
% Implementation of Garrett et al. (2014) 
% INPUTS:
% threshold: threshold to get visual field patch from the sign map (eq.2 of
% Garrett)
% structuringElementRadius: size of morphorogical opening after thresholding
% structuringElementRadius2: size of morphorogical closing after morphological opening
% 
% to show the resulting border:
% contour(im_final, 'k');

debug = false;
doImclose = false;

	if nargin == 1
		threshold = 0.75;
		structuringElementRadius = 3.0; 
        structuringElementRadius2 = 5.0;
    end

	std_signMap = nanstd(signMap(:));%std2(signMap);
    signMap(isnan(signMap))=0;
	signMapThreshold = signMap;

	multFactor = threshold;
	signMapThreshold(abs(signMapThreshold) < multFactor*std_signMap) = 0;
	signMapThreshold(signMapThreshold > multFactor*std_signMap) = 1;
	signMapThreshold(signMapThreshold < -multFactor*std_signMap) = -1;

	structuringElement = strel('disk',structuringElementRadius);
	structuringElement2 = strel('disk',structuringElementRadius2);

	signMapThresholdO = abs(signMapThreshold);
    
    if debug; figure;subplot(331); imagesc(signMapThresholdO); axis equal tight; title('abs(signmap threshold)'); end

	signMapThresholdO = imopen(signMapThresholdO,structuringElement);  %morphological opening
	signMapThresholdO = signMapThresholdO.*signMapThreshold;

	signMapBinary = abs(signMapThresholdO);
    if debug; subplot(332); imagesc(signMapThresholdO); axis equal tight; title('imopen'); end

    % % test 1: imreconstruct on thresholded signMap
    % Ie = imerode(signMapThresholdO,structuringElement);
    % Iobr = imreconstruct(Ie, signMapThreshold);
    % %Iobr == signMapThreshold
    % 
    % % test 2: imreconstruct on signMap
    % Ie = imerode(signMap,structuringElement);
    % Iobr = imreconstruct(Ie, signMap);
    % %difference between Iobr and signMap is negligible

	%Visual cortex border = Dilate(Open(Close(abs(Sthresh)))
	temp = imclose(signMapBinary,structuringElement2);
    if debug; subplot(333); imagesc(temp); axis equal tight; title('imclose'); end
	temp2 = imopen(temp,structuringElement2);
    if debug; subplot(334); imagesc(temp2); axis equal tight; title('imopen again');end
	visualCortexBorder = imdilate(temp2,structuringElement2);	%supposed to join boundaries of visual cortices
	visualCortexBorder = imfill(visualCortexBorder);
    if debug; subplot(335); imagesc(visualCortexBorder); axis equal tight; title('imdilate and imfill');end


	%border = visualCortexBorder - signMapBinary;
    border = visualCortexBorder - imerode(signMapThreshold<0, strel('disk',1)) - imerode(signMapThreshold>0, strel('disk',1));
    if debug; subplot(336); imagesc(border); axis equal tight; title('border by imerode');end
	border = abs(border);
    
    if doImclose
    	border = imclose(border, structuringElement);
        if debug; subplot(337); imagesc(border); axis equal tight; title('imclose'); end
    end

    border = bwmorph(border,'thin',Inf);
    if debug; subplot(338); imagesc(border); axis equal tight; title('thin'); end
	border = bwmorph(border,'spur',1);
    if debug; subplot(339); imagesc(border); axis equal tight; title('spur');end

	im = bwlabel(1-border,4);
	im(find(im == 1)) = 0;

	im_label = bwlabel(im,4);
	im_final = zeros(size(im_label));
	for l=1:length(unique(im_label))
		im_label_l = im_label;
		im_label_l(im_label_l~=l) = 0;
		im_sign = sign(mean2(im_label_l.*(signMapThresholdO)));
		im_label_l = ((im_label_l)./l)*im_sign;
		im_final = im_final + im_label_l;
	end

end