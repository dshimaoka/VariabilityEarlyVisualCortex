function subject_id = getSubjectId(loadFile)
if nargin < 1
    loadFile = '/home/daisuke/Documents/git/VariabilityEarlyVisualCortex/data/cifti_polarAngle_all.mat';
end

load(loadFile);
fields = fieldnames(cifti_polarAngle);    
subject_id = cellfun(@(x)sscanf(x,'x%6c_fit1_polarangle_msmall'), fields, 'UniformOutput', false);
noempty = cellfun(@(x)~isempty(x), subject_id);
subject_id = subject_id(noempty)';