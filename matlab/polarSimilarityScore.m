function similarity = polarSimilarityScore(pol, pol_ref, tolerance, minNPix, mask)
%similarity = polarSimilarityScore(pol, pol_ref, tolerance, minNPix, mask)
% OUTPUT:
% 1st entry: ROI
% 1: dorsal
% 2: ventral
% 3: both
%
% 2nd entry: similarity metric
% 1: bfscore
% 2: jaccard
% 3: dice
% 4: 1/chamferDistance

for itype = 1:3
    switch itype
        case 1 %ventral
            pol_binary = getBinaryTheta(interpNanImages(-pol), tolerance, minNPix, mask);
            pol_binary_ref = getBinaryTheta(interpNanImages(-pol_ref), tolerance, minNPix, mask);
            case 2 %dorsal
            pol_binary = getBinaryTheta(interpNanImages(pol), tolerance, minNPix, mask);
            pol_binary_ref = getBinaryTheta(interpNanImages(pol_ref), tolerance, minNPix, mask);
        case 3 %both
            pol_binary = logical(getBinaryTheta(interpNanImages(-pol), tolerance, minNPix, mask) ...
                + getBinaryTheta(interpNanImages(pol), tolerance, minNPix, mask));
            pol_binary_ref = logical(getBinaryTheta(interpNanImages(-pol_ref), tolerance, minNPix, mask) ...
                + getBinaryTheta(interpNanImages(pol), tolerance, minNPix, mask));
    end
        
    %% image similarity
    similarity(itype, 1) = bfscore(pol_binary, pol_binary_ref);
    similarity(itype, 2) = jaccard(pol_binary, pol_binary_ref);
    similarity(itype, 3) = dice(pol_binary, pol_binary_ref);
    similarity(itype, 4) = 1/chamferDistance(pol_binary, pol_binary_ref);

    % stats_ref = regionprops(pol_binary_ref, 'ConvexHull');
    % pol_poly_ref = polyshape(stats_ref(1).ConvexHull);
    % stats = regionprops(pol_binary, 'ConvexHull');
    % pol_poly = polyshape(stats(1).ConvexHull);
    % similarity(itype, 5) = turningdist(pol_poly, pol_poly_ref);
end