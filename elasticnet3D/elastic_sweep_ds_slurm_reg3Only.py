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



all_ids = ['avg'] #['114823','157336','585256','581450','725751','avg']; #from Ribeiro 2023 Fig1
loadDir = '/mnt/dshi0006_market/VariabilityEarlyVisualCortex/';


for ids in range(0,len(all_ids)):
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
    fixIdx = [] #area(s) for reference
    varIdx = [0,1,2] #area(s) for elastic net simulation
    
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
        mask_fix_idx = [];#np.concatenate([mask_idx[index] for index in fixIdx])
        mask_var_idx = np.concatenate([mask_idx[index] for index in varIdx])
    elif tgt == "D":
        mask_fix_idx = np.concatenate([mask_d_idx[index] for index in fixIdx])
        mask_var_idx = np.concatenate([mask_d_idx[index] for index in varIdx])
    elif tgt == "V":
        mask_fix_idx = np.concatenate([mask_v_idx[index] for index in fixIdx])
        mask_var_idx = np.concatenate([mask_v_idx[index] for index in varIdx])
    
#    a,b = dst.ind2sub(shape2d, mask_fix_idx)
    mask_fix_sub = []; #np.column_stack((a,b)).astype(int) #[y,x]
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
    # beta3: regularization within variable pixels
    beta1 = tf.placeholder(tf.float64, shape=(), name="b1")
    beta2 = tf.placeholder(tf.float64, shape=(), name="b2")
    beta3 = tf.placeholder(tf.float64, shape=(), name="b3")
    
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
    


  
    def getRegTerm3(distance2D):
        #extract subscripts used 
        distance2D_tf_c = np.zeros((len(mask_var_idx),len(mask_var_idx)))
        for i in range(0,len(mask_var_idx)):
            for j in range(0,len(mask_var_idx)):
                distance2D_tf_c[i,j] = distance2D[np.where(gridIdx == mask_var_idx[i])[0][0],
                                        np.where(gridIdx == mask_var_idx[j])[0][0]]
        
        distance2D_tf = tf.constant(distance2D_tf_c)
        
        #src_idx = np.arange(0,len(mask_var_idx))
        yyb_diff = tf.expand_dims(y, 1) - tf.expand_dims(y, 0)
        yyb_normsq = tf.einsum('ijk,ijk->ij', yyb_diff, yyb_diff) # closeness in vf
        
        # strategy5: weighted by 1/exp(distance2D)
        distance2D_weighted = tf.multiply(1/tf.exp(distance2D_tf), yyb_normsq)
        reg3 = tf.reduce_sum(distance2D_weighted)
        return reg3
 
    reg3 = getRegTerm3(distance2D)
  
    
    #########################
    # optimization
    #########################
    global_step = tf.Variable(0, trainable=False)
    current_eta = tf.train.exponential_decay(eta0, global_step, 10, 0.99, staircase=False)
    
    
    
    def train(sess, opt, kappa, beta1, beta3, y, b1, b3, subject_id):
        sess.run(tf.global_variables_initializer())
        k = 30.0
        nIter = 1000  #1000
        reg1_all = np.zeros((nIter,1))
        reg3_all = np.zeros((nIter,1))
        for i in range(nIter):
            sess.run(opt, {kappa: k, beta1: b1, beta3: b3})
            k = k - k*5/nIter #0.005
            #print(sess.run(reg2))
            reg1_all[i] = sess.run(reg1)
            reg3_all[i] = sess.run(reg3)
                
        yfinal = sess.run(y)
        return yfinal, reg1_all, reg3_all


        

    ###########################
    # run simulation in (beta1, beta2) space
    ###########################
        
       
    numb1 = 5;
    numb3 = 5;
      
    for i1 in range(0, numb1):
        for i3 in range(0, numb3):
                #b1 = 0.02*1.6**i1 #smoothness
                #b2 = 0.02*1.6**i2 #inter-areal path length
                b1 = 0.01*2**i1
                b3 = 0.01*2**i3
                
                print('running elastic net b1:' + str(b1) + ', b3:' + str(b3))
                
                ## Minimal path length
                suffix = tgt + '_' + subject_id + '_b1_' + "%d"%(1e3*b1) + '_b3_' + "%d"%(1e3*b3)
                opt = tf.train.RMSPropOptimizer(current_eta, momentum = m).minimize(yx_cost + beta1 * reg1 + beta3 * reg3, global_step = global_step)
                result = train(sess, opt, kappa, beta1, beta3, y, b1, b3, subject_id)
                result2d = e2d.getResult2D(result[0], yb, mask_var_sub, mask_fix_sub, map_h, map_w)
                summary = e2d.resultSummary(result[0], yb, retinotopy, map_h, map_w, mask_idx, mask_var_idx, mask_var_sub, mask_fix_idx, mask_fix_sub, thisDir, suffix, subject_id)
          

          
