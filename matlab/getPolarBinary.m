function pol_binary = getPolarBinary(pol, itype, tolerance, minNPix, mask)

switch itype
    case 1
        pol_binary = getBinaryTheta(interpNanImages(-pol), tolerance, minNPix, mask);
    case 2
        pol_binary = getBinaryTheta(interpNanImages(pol), tolerance, minNPix, mask);
    case 3
        pol_binary = logical(getBinaryTheta(interpNanImages(-pol), tolerance, minNPix, mask) ...
            + getBinaryTheta(interpNanImages(pol), tolerance, minNPix, mask));
end
