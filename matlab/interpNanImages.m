function im_interp = interpNanImages(im, interpMethod)
% im_interp = interpNanImages(im);
% return image(s) where nans are filled

if nargin < 2
    interpMethod = 'nearest';
        %ok 'nearest','v4'
    %ng 'linear' 'natural' cubic'

end

im_interp = zeros(size(im));
for iii = 1:size(im,3)
    okIdx = ~isnan(im(:,:,iii));
    [yok,xok]=find(okIdx);
    [yng,xng]=find(isnan(im(:,:,iii)));
    
    tmp = reshape(im(:,:,iii),numel(im(:,:,iii)),1);
    zng=griddata(xok, yok, tmp(okIdx), xng, yng, interpMethod);
    test =  im(:,:,iii);
    for jjj = 1:numel(zng)
        test(yng(jjj),xng(jjj))=zng(jjj);
    end
    im_interp(:,:,iii) = test;
end
