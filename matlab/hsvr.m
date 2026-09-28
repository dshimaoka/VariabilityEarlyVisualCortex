function cmap = hsvr(n)
%for polar angle map ranging from -180 to 180 deg, where 

if nargin < 1
n = 256;
end

% HSV colormap
cmap = flipud(hsv(n));

% Shift so cyan (hue = 0.5) is at the beginning/end
cmap = circshift(cmap, round(n/2 - n/12));

% colormap(cmap);