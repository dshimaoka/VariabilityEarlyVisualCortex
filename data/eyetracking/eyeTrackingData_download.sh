#!/bin/bash

subjectlist=./../../list_subj.txt

while read -r subject;
do
    aws s3 cp \
        s3://hcp-openaccess/HCP_1200/$subject/unprocessed/7T/tfMRI_RETBAR1_AP/LINKED_DATA/EYETRACKER/"$subject"_7T_RETBAR1_eyetrack.asc \
        .
done < $subjectlist