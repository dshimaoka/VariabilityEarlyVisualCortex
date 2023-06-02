% This code performs eye-tracking data processing and analysis. It extracts 
% fixation events from eye-tracking data files and calculates the mean 
% deviation of the fixations in the x and y axes. It then saves the mean 
% deviation values for each participant and task in a structure called 
% meanDeviationData.
% 
% The code follows these main steps:
% 
%     Clearing the workspace and command window.
%     Creating a structure meanDeviationData to store the mean deviation values.
%     Listing the files in a specific folder (replace 'folder_path' with the actual path).
%     Creating a temporary working folder for extracting text files.
%     Looping through the files and processing each file individually.
%     Loading and processing eye-tracking data from ASC files.
%     Extracting fixation events, saccade events, blink events, and trigger events from the data.
%     Storing the extracted data in the dataEyelink structure.
%     Cleaning up temporary files.
%     Finding fixations for each trial/task based on trigger values.
%     Calculating the mean deviation in x and y axes for each fixation.
%     Storing the mean deviation values in the meanDeviationData structure.
%     Saving the meanDeviationData structure to a MAT file named meanDeviationData.mat.
%     Calculating the average values for each participant from the meanDeviationData structure.
%     Storing the average values in the participantAverages structure.
%     Converting the participantAverages structure into a table, removing the 'P' prefix from participant IDs.
%     Saving the table as a CSV file named eyetrackingParticipantAveragesXY.csv.
%     Displaying a message confirming the successful save.
%     Plotting the deviation data in a scatter plot.
% 
% Please note that the code assumes the availability of specific input files 
% in the given folder path and performs various data processing operations 
% using shell commands (e.g., using sed to extract specific data). Ensure 
% that the input files and folder structure are appropriately set up before 
% running the code.
% 
% NB above symmary from chatGPT
% NB system functions adapted from Jian Chen
% (https://jianchen.info/images/eyelink/eyelinkpub.html)
%%
clc; clear all; close all;


% Create a structure to store the mean deviation values]]
meanDeviationData = struct();

% List of files in the folder
file_list = dir('./../data/eyetracking/*.asc');  % Replace 'folder_path' with the actual folder path containing the files

% make a working folder for .txt extraction
if ~exist('./../data/eyetracking/eyetrackingTmp/', 'dir')
    mkdir('./../data/eyetracking/eyetrackingTmp/');
end
cd('./../data/eyetracking/eyetrackingTmp/');


% Loop through the files
for i = 1:numel(file_list)
    disp(i)

    % Extract participant and task information from the file name
    file_name = file_list(i).name;
    parts = strsplit(file_name, '_');
    participant = ['P', num2str(parts{1})];
    task = parts{3};
    
    % Load and process the data
    filenameASC = fullfile(file_list(i).folder, file_list(i).name);
    % read in the data, deal with ppts whose conversion have gone awry with
    % letters in the fields (remove all alphabetical characters)
    system(['sed -n ''/^[0-9]/p'' ' filenameASC ' | tr -d [:alpha:] | sed ''s/ \./0.0/g'' | sed ''s/\.//g'' > tmp_samp.txt']); 
    % read samples into data structure COL 2, 3
    dataEyelink.samples = dlmread('tmp_samp.txt');
    dataEyelink.samples(:,2:3) = dataEyelink.samples(:,2:3)/10; % correct for above hack to resote pixel values....
    
    % extract fixation events from left eye
    % again deal with instances where no data was recorded (i.e. blinks: . have been turned into 0.0)
    % x and y locations for fixations are in columns 4, 5
    system(['sed -n -e ''/^EFIX/p'' ' filenameASC ' | sed ''s/EFIX L   //g'' | sed -n ''/^[0-9]/p'' | sed ''s/ \./0.0/g'' > tmp_fix.txt']);
    fixation_file = 'tmp_fix.txt';
    tmpStruct = dir(fixation_file);
    if ~exist(fixation_file, 'file') || exist(fixation_file, 'file') >= 1 && tmpStruct.bytes == 0
        fprintf('Fixation file is empty or does not exist. Skipping to next file.\n');
        continue; % Skip to next iteration of the loop
    else
        dataEyelink.fixations = dlmread(fixation_file);
    end
    
    % extract saccade events: COL FROM(X) 4, TO 6; FROM(Y) 5 TO 7
    % again deal with instances where no data was recorded (i.e. blinks: . have been turned into 0.0)
    system(['sed -n -e ''/^ESACC/p'' ' filenameASC ' | sed ''s/ESACC L  //g'' | sed -n ''/^[0-9]/p'' | sed ''s/ \./0.0/g'' > tmp_sacc.txt']);
    sacc_file = 'tmp_sacc.txt';
    tmpStruct = dir(sacc_file);
    if ~exist(sacc_file, 'file') || exist(sacc_file, 'file') >= 1 && tmpStruct.bytes == 0
        fprintf('Saccade file is empty or does not exist. Skipping to next file.\n');
        continue; % Skip to next iteration of the loop
    else
        dataEyelink.saccades = dlmread('tmp_sacc.txt');
    end
    
    % extract blink events
    system(['sed -n -e ''/^EBLINK/p'' ' filenameASC ' | sed ''s/EBLINK L //g'' > tmp_blnk.txt']);
    % read blinks into data structure
    dataEyelink.blinks = dlmread('tmp_blnk.txt');
    
    % extract trigger event for start and end of 'trial'
    % Read the contents of the file
    fileContents = fileread(filenameASC);
    
    % Extract the trigger values using regular expressions
    inputTrigger = regexp(fileContents, '^START\s+(\d+)', 'tokens', 'lineanchors');
    endTrigger = regexp(fileContents, '^END\t([0-9]+)', 'tokens', 'lineanchors');
    
    % Check if trigger values were found
    if ~isempty(inputTrigger) && ~isempty(endTrigger)
        inputTriggerValue = str2double(inputTrigger{1}{1});
        endTriggerValue = str2double(endTrigger{1}{1});
        
        % Save the trigger values to a text file
        fid = fopen('tmp_trgr.txt', 'w');
        fprintf(fid, '%d\n%d\n', inputTriggerValue, endTriggerValue);
        fclose(fid);
        
        % read triggers into data structure
        dataEyelink.triggers = dlmread('tmp_trgr.txt');
    else
        fprintf('Trigger values not found in the file.\n');
        continue; % Skip to next iteration of the loop
    end
    
    % clean up after this ppt/task
    system('rm tmp_samp.txt');
    system('rm tmp_fix.txt');
    system('rm tmp_sacc.txt');
    system('rm tmp_blnk.txt');
    system('rm tmp_trgr.txt');
    
    %we care about fixations so find those (which happen within start to
    %finish of the 'trial'/task)
    %find fixations for each trial
    for j = 1:2:length(dataEyelink.triggers)
        res.eye.fixations{round(j/2)} = dataEyelink.fixations(...
            dataEyelink.triggers(j,1) < dataEyelink.fixations(:,1) & ...
            dataEyelink.fixations(:,1) < dataEyelink.triggers(j+1,1),:);
    end
    
    % Define screen dimensions
    screenWidth = 1024;
    screenHeight = 768;
    
    % Define reference point
    refPoint = [screenWidth/2, screenHeight/2];  % Assuming the reference point is at the center of the screen
    
    % Calculate deviation for each fixation
    deviationFix = res.eye.fixations{1, 1}(:, 4:5) - refPoint;
    
    % Calculate mean deviation in x and y axes
    meanDeviationXFix = mean(abs(deviationFix(:,1)));
    meanDeviationYFix = mean(abs(deviationFix(:,2)));
    
    % Store mean deviation values in the structure
    meanDeviationData.(participant).(task).meanDeviationX = meanDeviationXFix;
    meanDeviationData.(participant).(task).meanDeviationY = meanDeviationYFix;
end

% Save the structure to a MAT file
save('meanDeviationData.mat', 'meanDeviationData');

%% get the averages for all ppts

% Initialize a structure to store the average values for each participant
participantAverages = struct();

% Iterate over the participant IDs
participantIDs = fieldnames(meanDeviationData);
for i = 1:numel(participantIDs)
    participantID = participantIDs{i};
    participantData = meanDeviationData.(participantID);
    
    % Initialize variables to accumulate the sum and count for each participant
    sumDeviationX = 0;
    sumDeviationY = 0;
    count = 0;
    
    % Iterate over the tasks for each participant
    tasks = fieldnames(participantData);
    for j = 1:numel(tasks)
        task = tasks{j};
        taskData = participantData.(task);
        
        % Extract the mean deviation x and y values for each task
        deviationX = taskData.meanDeviationX;
        deviationY = taskData.meanDeviationY;
        
        % Accumulate the sum
        sumDeviationX = sumDeviationX + deviationX;
        sumDeviationY = sumDeviationY + deviationY;
        
        % Increment the count
        count = count + 1;
    end
    
    % Calculate the average values for the participant
    averageDeviationX = sumDeviationX / count;
    averageDeviationY = sumDeviationY / count;
    
    % Store the average values in the participantAverages structure
    participantAverages.(participantID).averageDeviationX = averageDeviationX;
    participantAverages.(participantID).averageDeviationY = averageDeviationY;
end

%% put the new average structure into a table and remove the P from the ppt ID
% Initialize empty arrays for participant ID, deviation x, and deviation y
participantID = [];
deviationX = [];
deviationY = [];

% Extract data from the structure and store it in arrays
participantIDs = fieldnames(participantAverages);
for i = 1:numel(participantIDs)
    % Get the participant ID
    currParticipantID = participantIDs{i};
    
    % Remove 'P' from participant ID
    currParticipantIDnew = strrep(currParticipantID, 'P', '');
    
    % Get the deviation x and deviation y values
    currDeviationX = participantAverages.(currParticipantID).averageDeviationX;
    currDeviationY = participantAverages.(currParticipantID).averageDeviationY;
    
    % Append the values to the respective arrays
    participantID = [participantID; currParticipantIDnew];
    deviationX = [deviationX; currDeviationX];
    deviationY = [deviationY; currDeviationY];
end

% Create a table from the extracted data
participantTable = table(participantID, deviationX, deviationY);

%% save for Nanda
% Specify the file path and name
csvFileName = 'eyetrackingParticipantAveragesXY.csv';

% Save the table to a CSV file
writetable(participantTable, csvFileName, 'Delimiter', ',');

% Display a message indicating successful save
disp(['Table saved to ' csvFileName]);


%% have a look
% Plot the data
figure;
scatter(participantTable.deviationX, participantTable.deviationY);
xlabel('Deviation X');
ylabel('Deviation Y');
title('Participant Deviation Data');
