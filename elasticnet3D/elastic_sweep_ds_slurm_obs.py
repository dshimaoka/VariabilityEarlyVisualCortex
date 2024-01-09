# Juen 18 2020
# this is based on the sript that I gave declan
# upgraded to Python 3 and tensorflow 2

import os
jobid = os.getenv('SLURM_ARRAY_TASK_ID')

if jobid == None:
    sweep_id = 0;
else:
    sweep_id = int(jobid)
print(sweep_id)

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

import matplotlib.pyplot as plt
import functions.dstools as dst
import time

###########################
# parameters
###########################

np.random.seed(sweep_id)

out_root_dir = "/tmp/";#"/home/earsenau/sz11_scratch/elasticnet/"
tgt = "D"#"D"#"V+D"

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
#from compute_minimal_path_femesh.m
loadDir = '/home/daisuke/Documents/git/VariabilityEarlyVisualCortex/results/';
distance4D = scipy.io.loadmat(osp.join(loadDir, 'minimal_path_hmax2.mat'))['distance4D']
distance2D = scipy.io.loadmat(osp.join(loadDir, 'minimal_path_hmax2.mat'))['distance2D']

## load retinotopy and vfs
#from export_retinotopy.py
retinotopyData = scipy.io.loadmat(osp.join(loadDir, 'fieldSign_avg_smoothed'))
maltitude = retinotopyData['maltitude']
mazimuth = retinotopyData['mazimuth']

# from defineArealBorders.m
arealBorders = scipy.io.loadmat(osp.join(loadDir, 'fieldSign_avg_smoothed_arealBorder'))

shape2d = arealBorders['areaMatrix'][0][0].shape

final_mask_L_idx = retinotopyData['final_mask_L_idx'].astype(int)[0]
final_mask_L_d_idx = retinotopyData['final_mask_L_d_idx'].astype(int)[0]
a,b = dst.ind2sub(shape2d, final_mask_L_idx)
final_mask_L_sub = np.column_stack((a,b)).astype(int) #[y,x]

mask_v1_sub = np.argwhere(arealBorders['areaMatrix'][0][0] == 1) #[y,x]
mask_v2_sub = np.argwhere(arealBorders['areaMatrix'][0][1] == 1) #[y,x]
# mask_v2_sub = np.argwhere(arealBorders['areaMatrix'][0][1] + 
                            # arealBorders['areaMatrix'][0][2] == 1) #[y,x]
mask_v1_idx = dst.sub2ind(shape2d, mask_v1_sub[:,0], mask_v1_sub[:,1]) #[v1 pixels x 1]
mask_v2_idx = dst.sub2ind(shape2d, mask_v2_sub[:,0], mask_v2_sub[:,1])#[v2 pixels x 1]

#mask_v1_idx = np.nonzero((arealBorders['areaMatrix'][0][0]).flatten() == 1)[0] #identical to above


retinotopy = np.column_stack((mazimuth.flatten().T, maltitude.flatten().T))#[all pixels x 2]. 2nd argument is [azimuth altitude]

mask_v1_d_idx = np.intersect1d(mask_v1_idx, final_mask_L_d_idx)
mask_v2_d_idx = np.intersect1d(mask_v2_idx, final_mask_L_d_idx)
a,b = dst.ind2sub(shape2d, mask_v1_d_idx)
mask_v1_d_sub = np.column_stack((a,b)).astype(int) #[y,x]
a,b = dst.ind2sub(shape2d, mask_v2_d_idx)
mask_v2_d_sub = np.column_stack((a,b)).astype(int) #[y,x]

mask_v1_v_idx = np.setdiff1d(mask_v1_idx, final_mask_L_d_idx)
mask_v2_v_idx = np.setdiff1d(mask_v2_idx, final_mask_L_d_idx)
a,b = dst.ind2sub(shape2d, mask_v1_v_idx)
mask_v1_v_sub = np.column_stack((a,b)).astype(int) #[y,x]
a,b = dst.ind2sub(shape2d, mask_v2_v_idx)
mask_v2_v_sub = np.column_stack((a,b)).astype(int) #[y,x]

if tgt == "V+D":
    mask_fix_idx = mask_v1_idx
    mask_var_idx = mask_v2_idx
elif tgt == "D":
    mask_fix_idx = mask_v1_d_idx
    mask_var_idx = mask_v2_d_idx
elif tgt == "V":
    mask_fix_idx = mask_v1_v_idx
    mask_var_idx = mask_v2_v_idx

a,b = dst.ind2sub(shape2d, mask_fix_idx)
mask_fix_sub = np.column_stack((a,b)).astype(int) #[y,x]
a,b = dst.ind2sub(shape2d, mask_var_idx)
mask_var_sub = np.column_stack((a,b)).astype(int) #[y,x]


#########################
# set up the variables
#########################
n_prototypes = len(mask_v2_d_idx)
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

#option1: use V1
#pts = retinotopy[mask_fix_idx,:] #should use v2 instead??

#option1: use V2
pts = retinotopy[mask_var_idx,:] #should use v2 instead??

pts = pts[pts[:,0]>0] #altitude to be greater than 0
x0 = e2d.generate_landmarks_fromData(pts, n_prototypes, noise = prototype_noise)


# x0: [n_prototypes x 2]. 2nd argument is (azimuth, altitude)
x  = tf.constant(
        x0,
        dtype=tf.float64,
        name='x')


#### train these points on a cortical map (dimenion: map_h * map_w)
v2_len = len(mask_var_idx)
y0 = e2d.initial_condition(v2_len, x0)
# y0: [len(mask_v2) x 2]. 2nd argument is (azimuth, altitude)
y = tf.get_variable(
        "y",
        dtype = tf.float64,
        initializer = y0)

# visual field of V1 (fixed) == x??
#yb = y[:map_h]
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

#extract subscripts used 
distance2D_tf_c = np.zeros((len(mask_var_idx),len(mask_fix_idx)))
for i in range(0,len(mask_var_idx)):
    for j in range(0,len(mask_fix_idx)):
        distance2D_tf_c[i,j] = distance2D[np.where(final_mask_L_idx == mask_var_idx[i])[0][0],
                                np.where(final_mask_L_idx == mask_fix_idx[j])[0][0]]

distance2D_tf = tf.constant(distance2D_tf_c)

# distance1D_tf_c = distance2D_tf_c.flatten();
# distance1D_tf = tf.constant(distance1D_tf_c)



src_idx = np.arange(0,len(mask_var_idx))
yyb_diff = tf.expand_dims(y, 1) - tf.expand_dims(yb, 0)
yyb_normsq = tf.einsum('ijk,ijk->ij', yyb_diff, yyb_diff) # closeness in vf

## strategy3: for each V2 pixel, choose pixel in V1 whose path distance is shortest
# Find indices of the minimum values along each row
min_indices = np.argmin(distance2D_tf_c, axis=1)
# Create a new matrix of zeros with the same shape as the original matrix
distance2D_tf_masked_c = np.zeros_like(distance2D_tf_c)
rows = np.arange(distance2D_tf_c.shape[0])
distance2D_tf_masked_c[rows, min_indices] = np.min(distance2D_tf_c, axis=1)    
distance2D_tf_masked = tf.constant(distance2D_tf_masked_c)
distance2D_weighted = tf.multiply(distance2D_tf_masked, yyb_normsq)
reg2 = tf.reduce_sum(distance2D_weighted)

#sanity check
orig2d = np.nan * np.ones((map_h,map_w,2))
for pp in range(0,len(mask_var_idx)):
    orig2d[mask_var_sub[pp,1],mask_var_sub[pp,0],:] = retinotopy[mask_var_idx[pp],:]
for qq in range(0,len(mask_fix_idx)):
    orig2d[mask_fix_sub[qq,1],mask_fix_sub[qq,0],:] = retinotopy[mask_fix_idx[qq],:]
#plt.imshow(orig2d[:,:,0].T, origin='lower'); plt.colorbar(); plt.title('azimuth')
mask2d = ~np.isnan(orig2d[:,:,0])
idx = 100
plt.imshow(np.multiply(mask2d.T, distance4D[mask_var_sub[idx,0],mask_var_sub[idx,1],:,:]), origin='lower')
plt.plot(mask_var_sub[idx,1],mask_var_sub[idx,0],'rx')
plt.plot(mask_fix_sub[min_indices[idx],1], mask_fix_sub[min_indices[idx],0],'ro')
#distance4D looks all good
#min_indices is something wrong for dorsal V2
#, meaning something wrong at disatnce2D_tf  - something wrong when computing distance2D_tf from distance2D
# or distance2D? - something wrong for conversion from 4D to 2D


##FIXME: idex used in distance2D defined in matlab do NOT match idx defined in dstools
distance2D_test = distance2D[np.where(final_mask_L_idx == mask_var_idx[idx])[0][0],:]

distance2D_test2D = np.nan * np.ones((100,100))
for iii in range(0,len(final_mask_L_idx)):
    distance2D_test2D[final_mask_L_sub[iii,0],final_mask_L_sub[iii,1]] = distance2D_test[iii];#OK
plt.imshow(distance2D_test2D, origin='lower')
plt.plot(mask_var_sub[idx,1],mask_var_sub[idx,0],'rx')
plt.plot(mask_fix_sub[min_indices[idx],1], mask_fix_sub[min_indices[idx],0],'ro')

## strategy4: for each V1 pixel, choose pixel in V2 whose path distance is shortest

#########################
# optimization
#########################
global_step = tf.Variable(0, trainable=False)
current_eta = tf.train.exponential_decay(eta0, global_step, 10, 0.99, staircase=False)

cost = yx_cost + beta1 * reg1 + beta2 * reg2

opt = tf.train.RMSPropOptimizer(current_eta, momentum = m).minimize(cost, global_step = global_step)


def train(sess, opt, kappa, beta1, beta2, y, b1, b2, out_dir,reg2):
    sess.run(tf.global_variables_initializer())
    k = 30.0
    for i in range(1000): #1000
        sess.run(opt, {kappa: k, beta1: b1, beta2: b2})
        k = k - k*0.005
        print(sess.run(reg2))

    zz = sess.run(y)
    np.savetxt(out_dir + "y-" + "%6.5f"%b1 + '-' + "%6.5f"%b2 + ".data", zz)

###########################
# run simulation in (beta1, beta2) space
###########################


out_dir = out_root_dir + 'sweep_' + str(sweep_id).zfill(2) + "/"
if os.path.exists(out_dir) and os.path.isdir(out_dir):
    print("existing ", out_dir)        
else:
    print("creating ", out_dir)
    os.mkdir(out_dir)

sess = tf.Session()

for i1 in range(0, 1):
    for i2 in range(0, 1):
        b1 = 0.001*1.6**i1
        b2 = 0.01#4*1.6**i2
        print(b1, b2)
        train(sess, opt, kappa, beta1, beta2, y, b1, b2, out_dir, reg2)
        
        time.sleep(3)
        saveName = out_dir + "y-" + "%6.5f"%b1 + '-' + "%6.5f"%b2 + ".data"
        data = np.loadtxt(saveName)

        # result
        result2d = np.nan * np.ones((map_h,map_w,2))
        for pp in range(0,len(mask_var_idx)):
            result2d[mask_var_sub[pp,1],mask_var_sub[pp,0],:] = data[pp,:]
        for qq in range(0,len(mask_fix_idx)):
            result2d[mask_fix_sub[qq,1],mask_fix_sub[qq,0],:] = yb[qq,:]
        plt.subplot(121); plt.imshow(result2d[:,:,0].T, origin='lower'); plt.colorbar(); plt.title('azimuth')
        plt.subplot(122); plt.imshow(result2d[:,:,1].T, origin='lower'); plt.colorbar(); plt.title('altitude')
        plt.draw()
        
        # # initial condition
        # init2d = np.nan * np.ones((map_h,map_w,2))
        # for pp in range(0,len(mask_var_idx)):
        #     init2d[mask_var_sub[pp,1],mask_var_sub[pp,0],:] = y0[pp,:]
        # for qq in range(0,len(mask_fix_idx)):
        #     init2d[mask_fix_sub[qq,1],mask_fix_sub[qq,0],:] = yb[qq,:]
        # plt.subplot(121); plt.imshow(init2d[:,:,0].T, origin='lower'); plt.colorbar(); plt.title('azimuth')
        # plt.subplot(122); plt.imshow(init2d[:,:,1].T, origin='lower'); plt.colorbar(); plt.title('altitude')

        orig2d = np.nan * np.ones((map_h,map_w,2))
        for pp in range(0,len(mask_var_idx)):
            orig2d[mask_var_sub[pp,1],mask_var_sub[pp,0],:] = retinotopy[mask_var_idx[pp],:]
        for qq in range(0,len(mask_fix_idx)):
            orig2d[mask_fix_sub[qq,1],mask_fix_sub[qq,0],:] = retinotopy[mask_fix_idx[qq],:]
        plt.subplot(121); plt.imshow(orig2d[:,:,0].T, origin='lower'); plt.colorbar(); plt.title('azimuth')
        plt.subplot(122); plt.imshow(orig2d[:,:,1].T, origin='lower'); plt.colorbar(); plt.title('altitude')


    
