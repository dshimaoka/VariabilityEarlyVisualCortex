subject_id = {'157336','585256','114823','581450','725751'};


sid = 2;

% load(fullfile(loadDir, ['tri_faces_L_' subject_id{sid}]), 'tri_faces_L'); %outcome of data_formatting_individualSurface
% load(fullfile(loadDir, ['mid_pos_L_' subject_id{sid}]), 'mid_pos_L'); %outcome of data_formatting_individualSurface
% %    stlwrite(fullfile(saveDir, ['Geom_'  subject_id{sid} '.stl']), tri_faces_L, mid_pos_L)
% 
% surfaceMeshIn = surfaceMesh(single(tri_faces_L), single(mid_pos_L)); %NG
% surfaceMeshShow(surfaceMeshIn,Title="Original Mesh");

surfaceMeshOut = smoothSurfaceMesh(surfaceMeshIn,1);
%https://au.mathworks.com/help/lidar/ref/smoothsurfacemesh.html
%turned out to be smoothSurfaceMesh does NOT accecpt FEMesh

model = createpde(1);
importGeometry(model, fullfile(saveDir, ['Geom_'  subject_id{sid} '_' suffix '.stl'])); %"BracketTwoHoles.stl");%
generateMesh(model,"Hmax",hmax);%,"geometricOrder","linear","Hmin",0.2*mm); %determines coarseness of the mesh
