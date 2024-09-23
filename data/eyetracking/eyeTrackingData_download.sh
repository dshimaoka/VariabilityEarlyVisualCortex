#!/bin/bash

subjectlist=./../../list_subj.txt

while read -r subject;
do
    aws s3 cp \
        s3://hcp-openaccess/HCP_1200/$subject/unprocessed/7T/tfMRI_RETBAR1_AP/LINKED_DATA/EYETRACKER/"$subject"_7T_RETBAR1_eyetrack.asc \
        .
    aws s3 cp \
        s3://hcp-openaccess/HCP_1200/$subject/unprocessed/7T/tfMRI_RETBAR2_PA/LINKED_DATA/EYETRACKER/"$subject"_7T_RETBAR2_eyetrack.asc \
        .
    aws s3 cp \
        s3://hcp-openaccess/HCP_1200/$subject/unprocessed/7T/tfMRI_RETCCW_AP/LINKED_DATA/EYETRACKER/"$subject"_7T_RETCCW_eyetrack.asc \
        .
    aws s3 cp \
        s3://hcp-openaccess/HCP_1200/$subject/unprocessed/7T/tfMRI_RETCON_PA/LINKED_DATA/EYETRACKER/"$subject"_7T_RETCON_eyetrack.asc \
        .
    aws s3 cp \
        s3://hcp-openaccess/HCP_1200/$subject/unprocessed/7T/tfMRI_RETCW_PA/LINKED_DATA/EYETRACKER/"$subject"_7T_RETCW_eyetrack.asc \
        .
    aws s3 cp \
        s3://hcp-openaccess/HCP_1200/$subject/unprocessed/7T/tfMRI_RETEXP_AP/LINKED_DATA/EYETRACKER/"$subject"_7T_RETEXP_eyetrack.asc \
        .
done < $subjectlist