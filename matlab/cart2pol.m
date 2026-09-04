function [r, theta] = cart2pol(x,y)
 r = sqrt(x.^2 + y.^2);
 theta = 180/pi*atan2(y,x); % #[degree]
 % theta = shiftPA(180/pi*atan2(y,x)); % #[degree]
end

function z_values_PA = shiftPA(z_values_PA)
    % # # Shifting PA values so gist_rainbow_r colormap can be used
    % # sum = z_values_PA < 180
    % # minus = z_values_PA > 180
    % # z_values_PA[sum] = z_values_PA[sum] + 180
    % # z_values_PA[minus] = z_values_PA[minus] - 180
    % # return z_values_PA
    
    % Shifting PA values so gist_rainbow_r colormap can be used
    summation = z_values_PA < 0;
    %minus = z_values_PA > 180
    z_values_PA(summation) = z_values_PA(summation) + 360;
    %z_values_PA[minus] = z_values_PA[minus] - 180
end
