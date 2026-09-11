# add correlation values for a subject to one .csv file. 5x5 matrix
# find greatest correlation value for that subject, add to matrix with 'subject_id', 'b1', 'b2', 'pearson/spearman', 'azimuth/altitude/polar angle/eccentricity/circular correlation', 'correlation val'
#find max correlation val, plot scatter for this simulation vs empirical
# could return count for all maximum correlation values, determine which b1/b2 is most common

# check for patient id in directory
# if none, check for patient_id (done)
# if none, return "no 'results' folder"

# check 2d vs flat

# want to compare all correlation scores, finding the highest score for each patient, add this to a list, and then find the highest score of all the patients.
# preserve the b1 and b2 values to determine if there is a trend for a specific b1, or b2, or b1 + b2 combination
# add condition that checks for the patient id in the directory, if there is no matching folder, check for patient_id (done)
# additional condition to return "no 'result' folder"
# compute the correlation values for 2d vs flat as well

import os
import sys
os.chdir('/home/daisuke/Documents/git/VariabilityEarlyVisualCortex');
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import csv
import argparse
import numpy as np
import functions.dstools as dst
from scipy.stats import spearmanr
from pathlib import Path
import scipy.io as sio
import matplotlib.pyplot as plt
#os.chdir(r'c:\Users\61439\Variability_Early_Visual_Cortex_new\VariabilityEarlyVisualCortex') 
#loadDir = r'\\storage.erc.monash.edu\shares\MNHS-dshi0006\VariabilityEarlyVisualCortex';
loadDir = '/mnt/dshi0006_market/VariabilityEarlyVisualCortex/';

#iterating through each subject 

#add in a check or skip if the .mat file doesn't exist

#iterating through each combination of b1 and b2 values to load corresponding .mat files, extract correlation values, and save them to CSV files
#use commented code for all subjects once folder structure confirmed
# list_file = os.path.join(os.path.dirname(__file__), 'list_subj.txt')
# with open('list_subj.txt', 'r', encoding='utf-8') as f:
#     subject_id = [line.strip() for line in f if line.strip()]
subject_id = ['100610', '102311']

b1_b2_vals = [10]#, 20, 40, 80, 160]

for x in subject_id:
    thisDir = os.path.join(loadDir, x+'(done)')

    #load original retinotopy data for the subject
    arealBorders = sio.loadmat(os.path.join(thisDir, 'arealBorder_' + x + '.mat'))
    maltitude = arealBorders['grid_altitude_i'] 
    mazimuth = arealBorders['grid_azimuth_i']
    shape2d = arealBorders['areaMatrix'][0][0].shape
    retinotopy = np.column_stack((mazimuth.ravel(order='F'), maltitude.ravel(order='F')))

    nAreas = len(arealBorders['areaMatrix'][0])
    varIdx = [1, 2]
    
    ## computation of mask_var_idx    
    # mask_idx = [None] * nAreas
    # for iarea in range(nAreas):
    #     mask_idx[iarea] = np.where(arealBorders['areaMatrix'][0][iarea].ravel(order='F') == 1)[0]
    # mask_var_idx = np.concatenate([mask_idx[index] for index in varIdx])

    ## as in elastic_sweep_ds_slurm_all.py:
    mask_sub = [None]*nAreas
    mask_idx = [None]*nAreas
    for iarea in range(0, nAreas):
        mask_sub[iarea] = np.argwhere(arealBorders['areaMatrix'][0][iarea] == 1) #[y,x]
        mask_idx[iarea] = dst.sub2ind(shape2d, mask_sub[iarea][:,0], mask_sub[iarea][:,1]) #[v1 pixels x 1]
           
    mask_var_idx = np.concatenate([mask_idx[index] for index in varIdx])
      

    summary = sio.loadmat(os.path.join(thisDir, f'summary_correlation_{x}.mat'))

    # set up matrices
    # need to add corresponding b1/b2 values as titles
    spear_corr_altitude = np.zeros((5,5))
    spear_corr_azimuth = np.zeros((5,5))
    corr_pa = np.zeros((5,5))
    corr_ecc = np.zeros((5,5))

    for i in range(len(b1_b2_vals)):
        for j in range(len(b1_b2_vals)):
            b1 = b1_b2_vals[i]
            b2 = b1_b2_vals[j]
            
            print(f"Processing {x}, b1={b1}, b2={b2}")
            # Load the .mat file
            mat_file_path = os.path.join(thisDir, f'summary_{x}V+D_{x}_b1_{b1}_b2_{b2}.mat')
            if not os.path.exists(mat_file_path):
                print(f"MAT file not found for b1={b1}, b2={b2}, skipping.")
                continue
            mat_contents = sio.loadmat(mat_file_path)

            # Extract the results and pearson correlation coefficient values of the simulation
            result = mat_contents['result'][0,0]
            result = np.asarray(result)
            result = np.squeeze(result)
            
            # result = mat_contents['result'][0]
            # result = np.squeeze(result)

            # result2d = mat_contents['result2d']
            # result2d_flat = mat_contents['result2d_flat']

            # will create six 5x5 matrices for each patient_id

            #pearson correlation (cartesian)
            pear_corr_altitude = summary['corr_altitude'][i,j]
            pear_corr_azimuth = summary['corr_azimuth'][i,j]

            # do we need this circular corelation?
            # pear_corr_pa = summary['corr_pa'][i,j]

            #spearman correlation (cartesian)
            spear_corr_altitude[i,j] = spearmanr(retinotopy[mask_var_idx,1], result[:,1])[0]
            spear_corr_azimuth[i,j] = spearmanr(retinotopy[mask_var_idx,0], result[:,0])[0]
            #can't do circular spearman correlation for PA, so just report the pearson correlation for PA

            #convert data to polar
            #simulation data
            retinotopy_ecc, retinotopy_pa = dst.cartesian_to_polar(retinotopy[:,0],retinotopy[:,1])
            retinotopy_pol = np.column_stack((retinotopy_ecc,retinotopy_pa)) #[ecc, pa] in [deg]
            #empirical data
            result_ecc, result_pa = dst.cartesian_to_polar(result[:,0],result[:,1])

            #spearman correlation (polar)
            corr_pa[i,j] = spearmanr(np.pi/180*retinotopy_pol[mask_var_idx, 1], np.pi/180*result_pa).correlation
            corr_ecc[i,j] = spearmanr(retinotopy_pol[mask_var_idx, 0], result_ecc).correlation

            # on final iteration, we want to create several .csv files that tells us all correlation values, then determine maximum correlation value, and plot that
            if i==4 and j==4:
                # Save the correlation values to a CSV file
                os.makedirs('./results', exist_ok=True)
                row_col_labels = ["10", "20", "40", "80", "160"]
                # spearman correlation (altitude)
                csv_file_path = f'./results/spearcorr_{x}_altitude.csv'
                df = pd.DataFrame(spear_corr_altitude, index=row_col_labels, columns=row_col_labels)
                df.index.name = "B2/B1"
                df.to_csv(csv_file_path)
                print(f"Saved correlation values to {csv_file_path}")

                # spearman correlation (azimuth)
                csv_file_path = f'./results/spearcorr_{x}_azimuth.csv'
                df = pd.DataFrame(spear_corr_azimuth, index=row_col_labels, columns=row_col_labels)
                df.to_csv(csv_file_path)
                print(f"Saved correlation values to {csv_file_path}")

                # spearman correlation (polar angle)
                csv_file_path = f'./results/spearcorr_{x}_pa.csv'
                df = pd.DataFrame(corr_pa, index=row_col_labels, columns=row_col_labels)
                df.to_csv(csv_file_path)
                print(f"Saved correlation values to {csv_file_path}")

                # spearman correlation (eccentricity)
                csv_file_path = f'./results/spearcorr_{x}_ecc.csv'
                df = pd.DataFrame(corr_ecc, index=row_col_labels, columns=row_col_labels)
                df.to_csv(csv_file_path)
                print(f"Saved correlation values to {csv_file_path}")

                #determine max correlation value for all b1/b2 combinations, then plot this one.
                # max_pear_corr_altitude = np.max(pear_corr_altitude)
                # max_pear_corr_azimuth = np.max(pear_corr_azimuth)
                max_corr_altitude = np.max(spear_corr_altitude)
                max_corr_azimuth = np.max(spear_corr_azimuth)
                max_corr_pa = np.max(corr_pa)
                max_corr_ecc = np.max(corr_ecc)

                # max_idx_pear_altitude = np.unravel_index(np.argmax(pear_corr_altitude), pear_corr_altitude.shape)
                # max_idx_pear_azimuth = np.unravel_index(np.argmax(pear_corr_azimuth), pear_corr_azimuth.shape)
                max_idx_altitude = np.unravel_index(np.argmax(spear_corr_altitude), spear_corr_altitude.shape)
                max_idx_azimuth = np.unravel_index(np.argmax(spear_corr_azimuth), spear_corr_azimuth.shape)
                max_idx_pa = np.unravel_index(np.argmax(corr_pa), corr_pa.shape)
                max_idx_ecc = np.unravel_index(np.argmax(corr_ecc), corr_ecc.shape)

                # print(f"Max Correlations are: /n Pearson Altitude: {max_pear_corr_altitude} at b1={b1_b2_vals[max_idx_pear_altitude[0]]}, b2={b1_b2_vals[max_idx_pear_altitude[1]]} /n Pearson Azimuth: {max_pear_corr_azimuth} at b1={b1_b2_vals[max_idx_pear_azimuth[0]]}, b2={b1_b2_vals[max_idx_pear_azimuth[1]]} /n Spearman Altitude: {max_corr_altitude} at b1={b1_b2_vals[max_idx_altitude[0]]}, b2={b1_b2_vals[max_idx_altitude[1]]} /n Spearman Azimuth: {max_corr_azimuth} at b1={b1_b2_vals[max_idx_azimuth[0]]}, b2={b1_b2_vals[max_idx_azimuth[1]]} /n Spearman Polar Angle: {max_corr_pa} at b1={b1_b2_vals[max_idx_pa[0]]}, b2={b1_b2_vals[max_idx_pa[1]]} /n Spearman Eccentricity: {max_corr_ecc} at b1={b1_b2_vals[max_idx_ecc[0]]}, b2={b1_b2_vals[max_idx_ecc[1]]}")
                print(f"Max Correlations are: /n Spearman Altitude: {max_corr_altitude} at b1={b1_b2_vals[max_idx_altitude[0]]}, b2={b1_b2_vals[max_idx_altitude[1]]} /n Spearman Azimuth: {max_corr_azimuth} at b1={b1_b2_vals[max_idx_azimuth[0]]}, b2={b1_b2_vals[max_idx_azimuth[1]]}")


                # getting the indices for the max correlation values, then plotting the corresponding simulated data against the empirical data for that b1/b2 combination
                # corresponding indices
                max_i, max_j = max_idx_altitude

                # corresponding b1/b2 values
                max_b1 = b1_b2_vals[max_i]
                max_b2 = b1_b2_vals[max_j]

                # load matching result
                max_mat_path = os.path.join(thisDir,f'summary_{x}V+D_{x}_b1_{max_b1}_b2_{max_b2}.mat')
                max_mat = sio.loadmat(max_mat_path)
                max_result = max_mat['result'][0,0]
                max_result = np.asarray(max_result)
                max_result = np.squeeze(max_result)

                plt.figure(figsize=(12, 5))

                # azimuth
                plt.subplot(1, 2, 1)

                plt.scatter(retinotopy[mask_var_idx,0],max_result[:,0],alpha=0.5)

                plt.xlabel('Empirical Azimuth')
                plt.ylabel('Simulated Azimuth')

                plt.title(f'Subject {x} | b1={max_b1}, b2={max_b2}\n' f'Max corr = {max_corr_altitude:.3f}')

                # altitude
                plt.subplot(1, 2, 2)
                plt.scatter(retinotopy[mask_var_idx,1],max_result[:,1],alpha=0.5)

                plt.xlabel('Empirical Altitude')
                plt.ylabel('Simulated Altitude')

                plt.title(f'Subject {x} | b1={max_b1}, b2={max_b2}\n' f'Max corr = {max_corr_altitude:.3f}')
                plt.tight_layout()
                plt.savefig(f'./results/max_scatter_{x}_b1_{max_b1}_b2_{max_b2}.png',dpi=150)
                plt.close()

                # #plotting empirical vs simulated data for visual representation of the correlation
                # plt.figure(figsize=(12, 5))
                # #eccentricity
                # plt.subplot(1, 2, 1)
                # plt.scatter(retinotopy[mask_var_idx,0], result[:,0], alpha=0.5)
                # plt.xlabel('Empirical Azimuth')
                # plt.ylabel('Simulated Azimuth')
                # plt.title(f'Subject {x} - b1: {b1}, b2: {b2}')

                # #polar angle
                # plt.subplot(1, 2, 2)
                # plt.scatter(retinotopy[mask_var_idx,1], result[:,1], alpha=0.5)
                # plt.xlabel('Empirical altitude')
                # plt.ylabel('Simulated altitude')
                # plt.title(f'Subject {x} - b1: {b1}, b2: {b2}')
                # plt.title(f'Subject {x} - b1: {b1}, b2: {b2}')

                # #plotting empirical vs simulated data for visual representation of the correlation
                # plt.figure(figsize=(12, 5))
                # #eccentricity
                # plt.subplot(1, 2, 1)
                # plt.scatter(retinotopy_pol[mask_var_idx,0], result_ecc, alpha=0.5)
                # plt.xlabel('Empirical Eccentricity (deg)')
                # plt.ylabel('Simulated Eccentricity (deg)')
                # plt.title(f'Subject {x} - b1: {b1}, b2: {b2}')

                # #polar angle
                # plt.subplot(1, 2, 2)
                # plt.scatter(retinotopy_pol[mask_var_idx,1], result_pa, alpha=0.5)
                # plt.xlabel('Empirical Polar Angle (deg)')
                # plt.ylabel('Simulated Polar Angle (deg)')
                # plt.title(f'Subject {x} - b1: {b1}, b2: {b2}')
                # plt.title(f'Subject {x} - b1: {b1}, b2: {b2}')

                # plt.savefig(f'./results/scatter_{x}_b1_{b1}_b2_{b2}.png', dpi=150)
                # plt.close()