#!/bin/bash

export PWD=`pwd -P`
cd ./matlab/

git clone https://github.com/gllmflndn/gifti.git
cd gifti
git checkout cd1356fa6023acc6c9577e0e42ccf67aba9f4483
cd ./../

git clone https://github.com/fieldtrip/fieldtrip.git
cd fieldtrip
git checkout 6183b3ef726eff2bcc34aeb4252534bf7f21dbf8
cd ./../

singularity --silent exec  $neurodesk_singularity_opts --pwd $PWD /cvmfs/neurodesk.ardc.edu.au/containers/matlab_2022a_20221007/matlab_2022a_20221007.simg /opt/matlab/R2022a/bin/matlab -nodisplay -nodesktop -r "run data_formatting.m"

mv *.mat ../data/
cp ./S1200_7T_Retinotopy_Pr_9Zkk/S1200_7T_Retinotopy181/MNINonLinear/fsaverage_LR32k/*surf.gii ../data/

cd ./../
conda create -n VariabilityEarlyVisualCortex python=3.8
conda activate VariabilityEarlyVisualCortex

pip install -r requirements.txt
