
%subject_id = getSubjectId;
subject_id = {'134627', '155938', '193845', '200210', '318637'};%NG after hc laplacian twice
subject_id = {'155938','318637'};%NG after hc laplacian thrice
saveDir = '/mnt/dshi0006_market/VariabilityEarlyVisualCortex/';

type = 'midthickness';%'white' %cannot generateMesh with 'pial'
hmax = 2; %1: fine but too slow, 3: too coarse

ngsubject = [];
for sid= 1:numel(subject_id) 
 
    %% 1. import entire brain
    tic
    model = createpde(1);
    importGeometry(model, fullfile(saveDir, subject_id{sid}, ['Geom_'  subject_id{sid} '_hclaplacian.stl'])); %"BracketTwoHoles.stl");%
    %pdegplot(model)

    try
    generateMesh(model,"Hmax",hmax);%,"geometricOrder","linear","Hmin",0.2*mm); %determines coarseness of the mesh
    catch err
        ngsubject = [ngsubject subject_id{sid}];
        disp(subject_id{sid})
        disp(err)
    end
end