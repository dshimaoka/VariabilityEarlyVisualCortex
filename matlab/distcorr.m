function dcor = distcorr(x, y)
    % Validates column vectors
    x = x(:); y = y(:);
    n = length(x);
    
    % 1. Compute pairwise distance matrices
    A = pdist2(x, x);
    B = pdist2(y, y);
    
    % 2. Double-center the matrices
    rowMeanA = mean(A, 2); colMeanA = mean(A, 1); grandMeanA = mean(A, 'all');
    rowMeanB = mean(B, 2); colMeanB = mean(B, 1); grandMeanB = mean(B, 'all');
    
    A_centered = A - rowMeanA - colMeanA + grandMeanA;
    B_centered = B - rowMeanB - colMeanB + grandMeanB;
    
    % 3. Calculate distance covariance and variances
    dcov2 = sum(A_centered .* B_centered, 'all') / (n^2);
    dvarX2 = sum(A_centered .* A_centered, 'all') / (n^2);
    dvarY2 = sum(B_centered .* B_centered, 'all') / (n^2);
    
    % 4. Distance correlation
    if dvarX2 * dvarY2 > 0
        dcor = sqrt(dcov2 / sqrt(dvarX2 * dvarY2));
    else
        dcor = 0;
    end
end
