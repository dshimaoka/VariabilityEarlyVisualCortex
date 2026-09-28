function d = chamferDistance(A, B)

    A = logical(A);
    B = logical(B);

    % Distance to nearest foreground pixel
    dB = bwdist(B);
    dA = bwdist(A);

    % Symmetric Chamfer distance
    d1 = mean(dB(A));
    d2 = mean(dA(B));

    d = (d1 + d2) / 2;
end