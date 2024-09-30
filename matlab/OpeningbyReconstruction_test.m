%sophisticated image segmentation with imreconstruct and watershed
%https://au.mathworks.com/help/images/marker-controlled-watershed-segmentation.html

rgb = imread("pears.png");
I = im2gray(rgb);
imshow(I)

text(732,501,"Image courtesy of Corel(R)",...
     "FontSize",7,"HorizontalAlignment","right")


% step 2
gmag = imgradient(I);
imshow(gmag,[])
title("Gradient Magnitude")


% step 3
se = strel("disk",20);
Io = imopen(I,se);
imshow(Io)
title("Opening")


Ie = imerode(I,se);
Iobr = imreconstruct(Ie,I);
imshow(Iobr)
title("Opening-by-Reconstruction")

% Ioc = imclose(Io,se);
% imshow(Ioc)
% title("Opening-Closing")

Iobrd = imdilate(Iobr,se);
Iobrcbr = imreconstruct(imcomplement(Iobrd),imcomplement(Iobr));
Iobrcbr = imcomplement(Iobrcbr);
imshow(Iobrcbr)
title("Opening-Closing by Reconstruction")


fgm = imregionalmax(Iobrcbr);
imshow(fgm)
title("Regional Maxima of Opening-Closing by Reconstruction")


% I2 = labeloverlay(I,fgm);
% imshow(I2)
% title("Regional Maxima Superimposed on Original Image")

se2 = strel(ones(5,5));
fgm2 = imclose(fgm,se2);
fgm3 = imerode(fgm2,se2);

%step 4
bw = imbinarize(Iobrcbr);
imshow(bw)
title("Thresholded Opening-Closing by Reconstruction")


% step 5
gmag2 = imimposemin(gmag, bgm | fgm4);
L = watershed(gmag2);


% step 6
labels = imdilate(L==0,ones(3,3)) + 2*bgm + 3*fgm4;
I4 = labeloverlay(I,labels);
imshow(I4)
title("Markers and Object Boundaries Superimposed on Original Image")
