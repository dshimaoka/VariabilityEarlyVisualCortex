#!/bin/bash
#SBATCH --job-name=elasticnet
#SBATCH --mail-user=elizabeth.zavitz@monash.edu
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4096
#SBATCH --cpus-per-task=1


source /home/earsenau/envs/bin/activate
python3 /home/earsenau/elasticnet/test.py
