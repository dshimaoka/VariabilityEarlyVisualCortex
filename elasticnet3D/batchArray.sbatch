#!/bin/bash
#SBATCH --job-name=elasticnet
#SBATCH --mail-user=elizabeth.zavitz@monash.edu
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=7000
#SBATCH --cpus-per-task=12
#SBATCH --array=21-50

source /home/earsenau/envs/bin/activate
python /home/earsenau/elasticnet/elastic_sweep_slurm.py
