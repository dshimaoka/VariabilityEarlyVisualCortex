function [connectedPixels, connectedMatrix] = findConnectedPixels(matrix, label, targetPixels, flipy)
%[connectedPixels, connectedMatrix] = findConnectedPixels(matrix, label)
% select coordinates from GUI
%[connectedPixels, connectedMatrix] = findConnectedPixels(matrix, label, targetPixels)
% targetPixels(:,1) = x coordinates
% targetPixels(:,2) = y coordinates
if nargin < 4
    flipy = false;
end

if nargin < 3
    targetPixels = [];
end

if isempty(targetPixels)
    % Display the matrix as an image
    imagesc(matrix + 2); % Shifting the values to {1, 2, 3} for better visualization
    if flipy
        axis xy
    end
    %colormap('jet');
    colorbar;
    title(label);
    
    
    disp('Click on pixels then press Enter when done');
    %[x, y] = getpts;
    [targetPixels(:,1), targetPixels(:,2)] = getpts;
    if isempty(targetPixels)
        connectedPixels=[];
        connectedMatrix = nan(size(matrix));
        return;
    end
end
x = targetPixels(:,1);
y = targetPixels(:,2);

% Find connected pixels with the same value
clickedValue = [];
for i = 1:numel(x)
    clickedValue(i) = matrix(round(y(i)), round(x(i)));
end
[m, n] = size(matrix);
visited = false(m, n); % Track visited pixels
connectedPixels = [];

% Recursive function to find connected pixels
for i = 1:numel(x)
    findConnectedPixelsRecursive(round(y(i)), round(x(i)), clickedValue(i));
end

% Recursive function to find connected pixels
    function findConnectedPixelsRecursive(row, col, clickedValue)
        if row < 1 || row > m || col < 1 || col > n % Check for boundary conditions
            return;
        end
        
        if visited(row, col) % Check if the pixel is already visited
            return;
        end
        
        visited(row, col) = true; % Mark the pixel as visited
        
        % Check if the pixel has the same value as the clicked pixel
        if matrix(row, col) == clickedValue
            connectedPixels = [connectedPixels; [row, col]]; % Add the connected pixel coordinates
            % Recursively find connected pixels in the four neighboring directions
            findConnectedPixelsRecursive(row-1, col, clickedValue); % Up
            findConnectedPixelsRecursive(row+1, col, clickedValue); % Down
            findConnectedPixelsRecursive(row, col-1, clickedValue); % Left
            findConnectedPixelsRecursive(row, col+1, clickedValue); % Right
        end
    end

if ~isempty(connectedPixels)
    connectedMatrix = sub2mat(connectedPixels(:,2), ...
        connectedPixels(:,1), size(matrix));
end

end

function matrix = sub2mat(x,y,matrixSize)
matrix = zeros(matrixSize);
for idx = 1:numel(x)
    matrix(y(idx),x(idx)) = 1;
end
end
