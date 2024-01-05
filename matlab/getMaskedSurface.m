function [faces_mask, vertices_mask] = getMaskedSurface(faces, vertices, mask)

% Identify vertices associated with the masked subset of faces
vertices_in_mask_idx = find(mask);
vertices_mask= vertices(vertices_in_mask_idx,:);

% 3. Remove Faces Outside the Mask
% Identify faces that contain vertices outside the mask
faces_to_remove = [];
for i = 1:size(faces, 1)
    if ~all(ismember(faces(i, :), vertices_in_mask_idx))
        faces_to_remove = [faces_to_remove; i];
    end
end

% Remove identified faces
new_faces = setdiff(1:size(faces, 1), faces_to_remove);
updated_faces_obs_idx = faces(new_faces, :);
faces_mask = nan(size(updated_faces_obs_idx));
for ii = 1:size(updated_faces_obs_idx,1)
    for jj = 1:size(updated_faces_obs_idx, 2)
        faces_mask(ii,jj) = find(vertices_in_mask_idx == updated_faces_obs_idx(ii,jj));
    end
end

end
