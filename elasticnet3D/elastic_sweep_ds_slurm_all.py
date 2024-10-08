# September 27 2024
# Daisuke Shimaoka

import os
jobid = os.getenv('SLURM_ARRAY_TASK_ID')

if jobid == None:
    sweep_id = 0;
else:
    sweep_id = int(jobid)
#print(sweep_id)

############################
# elastic net with boundary
############################

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import os
os.chdir('/home/daisuke/Documents/git/VariabilityEarlyVisualCortex') 

import elasticnet3D.entools2D as e2d
import scipy.io
import os.path as osp

import functions.dstools as dst
#import time
import matplotlib.pyplot as plt
from scipy.io import savemat



all_ids = ['114823','157336','585256','581450','725751']; #from Ribeiro 2023 Fig1
loadDir = '/mnt/dshi0006_market/VariabilityEarlyVisualCortex/';


for ids in range(0,1):#len(all_ids)):
    subject_id = all_ids[ids]
    thisDir = osp.join(loadDir, subject_id);
    
    tf.reset_default_graph()
    sess = tf.Session()  # Create new session
    sess.run(tf.global_variables_initializer())
    
    ###########################
    # parameters
    ###########################
    
    np.random.seed(sweep_id)
    
    #out_root_dir = "/tmp/";#"/home/earsenau/sz11_scratch/elasticnet/"
    tgt = "V+D"#"D"#"V+D"
    fixIdx = [0] #area(s) for reference
    varIdx = [1,2] #area(s) for elastic net simulation
    
    ### learning parameters
    # the weights for the regularization terms
    # b1 - the weight for points inside the map
    # b2 - the weights for path distance
    # 0.01, 3
    #b1 = 0.01
    #b2 = 3
    
    eta0 = 0.05      # initial lerning rate
    m = 0.8         # momentum
    
    # add small mount of noise to the prototypes, which might give the solution some variations
    prototype_noise = False
    
    ## load human brain data 
    #from compute_minimal_path_femesh_individual.m
    distance2D = scipy.io.loadmat(osp.join(thisDir, 'minimal_path_midthickness_hmax2_' + subject_id + '.mat'))['distance2D']
    distance2D_euc = scipy.io.loadmat(osp.join(thisDir, 'minimal_path_midthickness_hmax2_' + subject_id + '.mat'))['distance2D_euc']
    distance2D_flat = scipy.io.loadmat(osp.join(thisDir, 'minimal_path_midthickness_hmax2_' + subject_id + '.mat'))['distance2D_flat']
    
    
    ## load retinotopy and vfs
    #from export_geometry_individual.py
    retinotopyData = scipy.io.loadmat(osp.join(thisDir, 'geometry_retinotopy_' + subject_id + '.mat'))#'fieldSign_avg_smoothed'))
    #maltitude = retinotopyData['grid_altitude'] #['maltitude']
    #mazimuth = retinotopyData['grid_azimuth']#['mazimuth']
    
    # from defineArealBorders_individual.m
    arealBorders = scipy.io.loadmat(osp.join(thisDir, 'arealBorder_' + subject_id + '.mat'))
    
    maltitude = arealBorders['grid_altitude_i'] 
    mazimuth = arealBorders['grid_azimuth_i']
    shape2d = arealBorders['areaMatrix'][0][0].shape
    gridIdx = np.arange(0,shape2d[0]*shape2d[1])
    
    final_mask_L_idx = retinotopyData['final_mask_L_idx'].astype(int)[0] #FIXME
    final_mask_L_d_idx = retinotopyData['final_mask_L_d_idx'].astype(int)[0] #FIXME
    a,b = dst.ind2sub(shape2d, final_mask_L_idx) #FIXME
    final_mask_L_sub = np.column_stack((a,b)).astype(int) #[y,x]  #FIXME
    
    retinotopy = np.column_stack((mazimuth.ravel(order='F'), maltitude.ravel(order='F')))#[all pixels x 2]. 2nd argument is [azimuth altitude]
    
    nAreas = len(arealBorders['areaMatrix'][0])
    mask_sub = [None]*nAreas
    mask_idx = [None]*nAreas
    mask_v_sub = [None]*nAreas
    mask_v_idx = [None]*nAreas
    mask_d_sub = [None]*nAreas
    mask_d_idx = [None]*nAreas
    for iarea in range(0, nAreas):
        mask_sub[iarea] = np.argwhere(arealBorders['areaMatrix'][0][iarea] == 1) #[y,x]
        mask_idx[iarea] = dst.sub2ind(shape2d, mask_sub[iarea][:,0], mask_sub[iarea][:,1]) #[v1 pixels x 1]
        mask_d_idx[iarea] = np.intersect1d(mask_idx[iarea], final_mask_L_d_idx)
        a,b = dst.ind2sub(shape2d, mask_d_idx[iarea])
        mask_d_sub[iarea] = np.column_stack((a,b)).astype(int) #[y,x]
        mask_v_idx[iarea] = np.setdiff1d(mask_idx[iarea], final_mask_L_d_idx)
        a,b = dst.ind2sub(shape2d, mask_v_idx[iarea])
        mask_v_sub[iarea] = np.column_stack((a,b)).astype(int) #[y,x]
    
    if tgt == "V+D":
        mask_fix_idx = np.concatenate([mask_idx[index] for index in fixIdx])
        mask_var_idx = np.concatenate([mask_idx[index] for index in varIdx])
    elif tgt == "D":
        mask_fix_idx = np.concatenate([mask_d_idx[index] for index in fixIdx])
        mask_var_idx = np.concatenate([mask_d_idx[index] for index in varIdx])
    elif tgt == "V":
        mask_fix_idx = np.concatenate([mask_v_idx[index] for index in fixIdx])
        mask_var_idx = np.concatenate([mask_v_idx[index] for index in varIdx])
    
    a,b = dst.ind2sub(shape2d, mask_fix_idx)
    mask_fix_sub = np.column_stack((a,b)).astype(int) #[y,x]
    a,b = dst.ind2sub(shape2d, mask_var_idx)
    mask_var_sub = np.column_stack((a,b)).astype(int) #[y,x]
    
    
    #########################
    # set up the variables
    #########################
    n_prototypes = len(mask_var_idx)
    map_h = shape2d[0] #nRows
    map_w = shape2d[1] #nCols
    
    # the weights of the regularization term
    # beta1: the standard beta
    # beta2: additional constraints on the boundary
    beta1 = tf.placeholder(tf.float64, shape=(), name="b1")
    beta2 = tf.placeholder(tf.float64, shape=(), name="b2")
    
    # annealing parameter
    kappa = tf.placeholder(tf.float64, shape=(), name="k")
    
    #### prototypes
    # generate prototypes on the visual field
    #x0 = e2d.generate_landmarks(magnification, ecc0, ecc1, n_prototypes, noise = prototype_noise)
    pts = retinotopy[mask_var_idx,:] #should use v2 instead??
    
    #pts = pts[pts[:,0]>0] #altitude to be greater than 0 #commented out 30/9/24
    x0 = e2d.generate_landmarks_fromData(pts, n_prototypes, noise = prototype_noise)
    
    # x0: [n_prototypes x 2]. 2nd argument is (azimuth, altitude)
    x  = tf.constant(
            x0,
            dtype=tf.float64,
            name='x')
    
    #### train these points on a cortical map (dimenion: map_h * map_w)
    y0 = e2d.initial_condition(len(mask_var_idx), x0)
    # y0: [len(mask_v2) x 2]. 2nd argument is (azimuth, altitude)
    y = tf.get_variable(
            "y",
            dtype = tf.float64,
            initializer = y0)
    
    # visual field of V1 (fixed) == x??
    yb = retinotopy[mask_fix_idx,:]
    
    #### main cost
    yx_diff = tf.expand_dims(y, 1) - tf.expand_dims(x, 0)
    yx_normsq = tf.einsum('ijk,ijk->ij', yx_diff, yx_diff)
    yx_gauss = tf.exp(-1.0 * yx_normsq / (2.0 * kappa * kappa))
    yx_cost = -1.0 * kappa * tf.reduce_sum(
                tf.log(
                    tf.reduce_sum(yx_gauss, axis=0)))
    
    #### regularization term 1 - within area
    # n is a list of map_h * map_w objects. The i-th item of n is a list containing the indices of nodes neighboring the i-th node
    n = e2d.neighborhood(map_h, map_w)
    mask = e2d.make_mask(map_h, map_w, n)
    #mask = tf.constant(e2d.make_mask(map_h, map_w, n), dtype=tf.float64, name='mask')
    
    # pairwise distance: first use broadcast to calculate pairwise difference
    yy_diff = tf.expand_dims(y, 1) - tf.expand_dims(y, 0)
    yy_normsq = tf.einsum('ijk,ijk->ij', yy_diff, yy_diff)
    yy_normsq_masked = tf.multiply(mask[mask_var_idx[:,np.newaxis], mask_var_idx], 
                                   yy_normsq)
    
    reg1 = tf.reduce_sum(yy_normsq_masked)
    
    #### regularization term 2 - path in cortex 
    # y: V2 position in visual field [azimuth altitude] (variable)
    # yb: V1 position in visual field [azimuth altitude] (fixed)
    
    def getRegTerm2(distance2D):
        #extract subscripts used 
        distance2D_tf_c = np.zeros((len(mask_var_idx),len(mask_fix_idx)))
        for i in range(0,len(mask_var_idx)):
            for j in range(0,len(mask_fix_idx)):
                distance2D_tf_c[i,j] = distance2D[np.where(gridIdx == mask_var_idx[i])[0][0],
                                        np.where(gridIdx == mask_fix_idx[j])[0][0]]
        
        distance2D_tf = tf.constant(distance2D_tf_c)
        
        #src_idx = np.arange(0,len(mask_var_idx))
        yyb_diff = tf.expand_dims(y, 1) - tf.expand_dims(yb, 0)
        yyb_normsq = tf.einsum('ijk,ijk->ij', yyb_diff, yyb_diff) # closeness in vf
        
        # strategy5: weighted by 1/exp(distance2D)
        distance2D_weighted = tf.multiply(1/tf.exp(distance2D_tf), yyb_normsq)
        reg2 = tf.reduce_sum(distance2D_weighted)
        return reg2

    
    reg2 = getRegTerm2(distance2D)

    # Euclidean distance on flat surface as a control
    reg2_flat = getRegTerm2(distance2D_flat)
    
    # Euclidean distance in 3D brain as another control
    reg2_euc = getRegTerm2(distance2D_euc)
    
    
    #########################
    # optimization
    #########################
    global_step = tf.Variable(0, trainable=False)
    current_eta = tf.train.exponential_decay(eta0, global_step, 10, 0.99, staircase=False)
    
    cost = yx_cost + beta1 * reg1 + beta2 * reg2
    
    opt = tf.train.RMSPropOptimizer(current_eta, momentum = m).minimize(cost, global_step = global_step)
    
    
    def train(sess, opt, kappa, beta1, beta2, y, b1, b2, reg2, subject_id):
        sess.run(tf.global_variables_initializer())
        k = 30.0
        nIter = 1000  #1000
        reg1_all = np.zeros((nIter,1))
        reg2_all = np.zeros((nIter,1))
        for i in range(nIter):
            sess.run(opt, {kappa: k, beta1: b1, beta2: b2})
            k = k - k*5/nIter #0.005
            #print(sess.run(reg2))
            reg1_all[i] = sess.run(reg1)
            reg2_all[i] = sess.run(reg2)
                
        yfinal = sess.run(y)
        return yfinal, reg1_all, reg2_all


        

    ###########################
    # run simulation in (beta1, beta2) space
    ###########################
        
       
    numb1 = 5;
    numb2 = 5;
    corr_azimuth = np.zeros((numb1,numb2))
    corr_altitude = np.zeros((numb1,numb2))
    corr_azimuth_flat = np.zeros((numb1,numb2))
    corr_altitude_flat = np.zeros((numb1,numb2))
    corr_azimuth_euc = np.zeros((numb1,numb2))
    corr_altitude_euc = np.zeros((numb1,numb2))

    for i1 in range(0, numb1):#0,5
        for i2 in range(0, numb2):#0,5
            #b1 = 0.02*1.6**i1 #smoothness
            #b2 = 0.02*1.6**i2 #inter-areal path length
            b1 = 0.01*2**i1
            b2 = 0.01*2**i2
            
            print('running elastic net b1:' + str(b1) + ', b2:' + str(b2))
            
            suffix = subject_id + '_b1_' + "%d"%(1e3*b1) + '_b2_' + "%d"%(1e3*b2)
            result = train(sess, opt, kappa, beta1, beta2, y, b1, b2, reg2, subject_id)
            
            summary = e2d.resultSummary(result[0], yb, retinotopy, map_h, map_w, mask_idx, mask_var_idx, mask_var_sub, mask_fix_idx, mask_fix_sub, thisDir, suffix, subject_id)
            corr_azimuth[i1,i2] = summary[0]
            corr_altitude[i1,i2] = summary[1]

            ## Euclidean distance on flat surface as a control
            #need a normalization factor for b2/reg2??
            suffix_flat = subject_id + '_b1_' + "%d"%(1e3*b1) + '_b2_' + "%d"%(1e3*b2) + "_flat"
            result_flat = train(sess, opt, kappa, beta1, beta2, y, b1, b2, reg2_flat, subject_id)
            
            summary_flat = e2d.resultSummary(result_flat[0], yb, retinotopy, map_h, map_w, mask_idx, mask_var_idx, mask_var_sub, mask_fix_idx, mask_fix_sub, thisDir, suffix_flat, subject_id)
            corr_azimuth_flat[i1,i2] = summary_flat[0]
            corr_altitude_flat[i1,i2] = summary_flat[1]

            ## Euclidean distance in 3D as a control
            #need a normalization factor for b2/reg2??
            suffix_euc = subject_id + '_b1_' + "%d"%(1e3*b1) + '_b2_' + "%d"%(1e3*b2) + "_euc"
            result_euc = train(sess, opt, kappa, beta1, beta2, y, b1, b2, reg2_euc, subject_id)
            
            summary_euc = e2d.resultSummary(result_euc[0], yb, retinotopy, map_h, map_w, mask_idx, mask_var_idx, mask_var_sub, mask_fix_idx, mask_fix_sub, thisDir, suffix_euc, subject_id)
            corr_azimuth_euc[i1,i2] = summary_euc[0]
            corr_altitude_euc[i1,i2] = summary_euc[1]

    #summary across simulations
    plt.subplot(321);
    plt.imshow(corr_azimuth, origin='lower'); plt.title('corr in azimuth'); plt.clim(.5, 1); plt.xlabel('b2: inter-areal path length'); plt.ylabel('b1: intra-areal smoothness')

    plt.subplot(322);
    plt.imshow(corr_altitude, origin='lower'); plt.title('corr in altitude'); plt.clim(.5, 1);
    
    plt.subplot(323);
    plt.imshow(corr_azimuth_flat, origin='lower'); plt.title('Euclidean dist on flat surface'); plt.clim(.5, 1); plt.xlabel('b2: inter-areal path length'); plt.ylabel('b1: intra-areal smoothness')

    plt.subplot(324);
    plt.imshow(corr_altitude_flat, origin='lower'); plt.clim(.5, 1);  
    
    plt.subplot(325);
    plt.imshow(corr_azimuth_euc, origin='lower'); plt.title('Euclidean dist in 3d brain'); plt.clim(.5, 1); plt.xlabel('b2: inter-areal path length'); plt.ylabel('b1: intra-areal smoothness')

    plt.subplot(326);
    plt.imshow(corr_altitude_euc, origin='lower'); plt.clim(.5, 1); plt.colorbar; 
    
    plt.draw()
    plt.gcf().set_size_inches(20, 15)
    plt.savefig(thisDir+'/summary_correlation_'+subject_id, dpi=200)
    plt.close()

    savemat(thisDir+'/summary_correlation_'+subject_id +'.mat', 
             {'corr_azimuth': corr_azimuth,'corr_altitude': corr_altitude, 
              'corr_azimuth_euc': corr_azimuth_euc,'corr_altitude_euc': corr_altitude_euc,
              'corr_azimuth_flat': corr_azimuth_flat,'corr_altitude_flat': corr_altitude_flat});
    
    