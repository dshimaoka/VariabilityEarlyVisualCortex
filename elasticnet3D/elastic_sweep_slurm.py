# Juen 18 2020
# this is based on the sript that I gave declan
# upgraded to Python 3 and tensorflow 2

import os
jobid = 1#os.getenv('SLURM_ARRAY_TASK_ID')


sweep_id = int(jobid)
print(sweep_id)

############################
# elastic net with boundary
############################

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
from scipy.special import erfinv
import os

###########################
# parameters
###########################

np.random.seed(sweep_id)

out_root_dir = "/tmp/";#"/home/earsenau/sz11_scratch/elasticnet/"

# map_h is the side that has a boundary
# this geometry is defined by neighborhood()
# for example, (h, w) = (5, 3)
# index:
#    x 0 5 10
#    x 1 6 11
#    x 2 7 12
#    x 3 8 13
#    x 4 9 14
# x is boundary

### cortex
boundary_len = 5.5  # in milimeter
map_h = 30          # this is on the boundary
map_w = 15

### visual field
# magnification has to be a negative numbers
# generate prototypes in the eccentricity range (ecc0, ecc1). They are in degrees
# is boundary is used, ecc1 essentially has to be 10.0
n_prototypes    = 500
magnification   = -0.4
ecc0            = 0.2
ecc1            = 10.0

### learning parameters
# the weights for the regularization terms
# b1 - the weight for points inside the map
# b2 - the weights for map boundary. If b2 is 0.0, no boundary condition
# 0.01, 3
b1 = 0.01
b2 = 3

eta0 = 0.05      # initial lerning rate
m = 0.8         # momentum

# add small mount of noise to the prototypes, which might give the solution some variations
prototype_noise = False

#############################
# v2 geometry
# specific to marmoset V2
#############################

# x is cortical space in mm
# it should go from 0.0mm to 8mm, which should give eccentricity to 0.8 to 10.0
# this is derived from the cortical magnification function of V2
# ie. first take the integral of the cortical magnification function, then make inverse function
# note: this is too much magnification. Use the version below
def x2ecc_old(x):
    return np.exp(1.7162 + 3.6761 * erfinv(-0.5449 + 0.092 * x))

# x is cortical space in mm
# it should go from 0.0mm to 5.5mm, which should give eccentricity to 2.0 to 10.0
# this is derived from the cortical magnification function of V2
# ie. first take the integral of the cortical magnification function, then make inverse function
def x2ecc(x):
    return np.exp(1.7162 + 3.6761 * erfinv(-0.306111 + 0.0923636 * x))

# generate the visual field coordinates of the rostral boundary of V2
# the length ogf the boundary is boundary_len mm,
# which is divided into div divisions
def generate_v2_boundary(boundary_len, div):
    zz = []
    for x in np.linspace(0.0, boundary_len, div):
        # horizontal meridian
        coords = [x2ecc(x), 0.0]
        zz.append(coords)
    return np.array(zz)

#############################
# gemoetry
#############################

### generate landmarks on the visual field
def spiral(c, r0, r1, n, shuffleP=True):
    """
        Make a spiral with n points.
        the distance from these points to (0, 0) ranges from r0 to r1
        the density of the dots falls off accoridng to r^c, where r is the distance to (0, 0)
        c should be a negative number
        returns a numpy array
    """
    def f(r):
        return np.sqrt(np.power(r, 1.0-c)/(1.0-c))

    n0 = np.power(r0 * r0 * (1.0-c), 1.0/(1.0-c))
    n1 = np.power(r1 * r1 * (1.0-c), 1.0/(1.0-c))

    lst = []
    for k in range(1, n+1):
        d = f( (n1-n0)/(n-1)*k + n0-(n1-n0)/(n-1) )
        theta = 3.1415926 * (3.0 - np.sqrt(5.0)) * (k-1.0)
        x = d * np.cos(theta)
        y = d * np.sin(theta)
        lst.append((x,y))
    res = np.asarray(lst, dtype=np.float64)
    if shuffleP:
        np.random.shuffle(res)
    return res

def generate_landmarks(c, r0, r1, n, noise=False):
    # generate landmarks on a hemifield
    # returns a numpy array

    # generate twice as many points because spiral() covers the entire visual field
    pts = spiral(c, r0, r1, 2*n)

    # boolean array, find those that are in the right hemifield
    pts_idx = pts[:, 0] >= 0.0

    # make a mask array
    pts_idx = np.stack([pts_idx, pts_idx])
    pts_idx = np.transpose(pts_idx)

    # keep points whose x coordinate is larger or equal to 0
    zz = np.extract(pts_idx, pts)
    # but since the extracted is flattend, I have to reshape it
    zz = zz.reshape([int(zz.shape[0]/2), 2])

    if noise:
        zz = zz + np.random.normal(0.0, 0.1, zz.shape)

    return zz

def neighborhood(h, w):
    """
        numbers {0...h*w-1} arranged in a hxw grid, produce a list of neighboring numbers.
        The ith element of the list is a list of numbers that neighbor i
    """
    # this line defines how the geometry is coded
    # for (h, w) = (4, 2)
    # 0 4
    # 1 5
    # 2 6
    # 3 7
    x = np.transpose(np.array(range(h*w)).reshape([w, h]))
    lst = []
    for i in range(w):
        for j in range(h):
            neighbors = []
            for ii in [-1, 0, 1]:
                for jj in [-1, 0, 1]:
                    if (i+ii>=0) and (i+ii<w) and (j+jj>=0) and (j+jj<h):
                        neighbors.append(x[j+jj,i+ii])
            lst.append(neighbors)
    return lst

def make_mask(h, w, neighbors):
    x = np.zeros([h*w, h*w], dtype=np.float64)
    for i in range(h*w):
        for j in neighbors[i]:
            x[i, j] = 1.0
    return x

def make_boundary_mask(n):
    """
        n: numer of elements on the boundary
        return a n*n array
    """
    x = np.zeros([n, n], dtype=np.float64)
    for i in range(n):
        for j in [-1, 0, 1]:
            if (i+j)>=0 and (i+j)<n:
                x[i, i+j] = 1.0
    return x

def initial_condition(h, w, x):
    """
        initialze a cortical map of the size [map_w * map_h]
        each point on the map is randomly assigned to prototypes on the visual field (x) with some added noise
    """
    map_size = h * w

    idx = np.random.randint(x.shape[0], size = map_size)
    y = x[idx, :] # randomly sample from x with replacement
    n = np.random.normal(0.0, 1.0, y.shape) # the standard deviation is set to 5.0 deg
    return y + n

#########################
# set up the variables
#########################

# the weights of the regularization term
# beta1: the standard beta
# beta2: additional constraints on the boundary
beta1 = tf.placeholder(tf.float64, shape=(), name="b1")
beta2 = tf.placeholder(tf.float64, shape=(), name="b2")

# annealing parameter
kappa = tf.placeholder(tf.float64, shape=(), name="k")

#### prototypes
# generate prototypes on the visual field
x0 = generate_landmarks(magnification, ecc0, ecc1, n_prototypes, noise = prototype_noise)
x  = tf.constant(
        x0,
        dtype=tf.float64,
        name='x')

# generate prototypes on the boundary
b0 = generate_v2_boundary(boundary_len, map_h)
b = tf.constant(
    b0,
    dtype=tf.float64,
    name='b')

#### train these points on a cortical map (dimenion: map_h * map_w)
y0 = initial_condition(map_h, map_w, x0)
y = tf.get_variable(
        "y",
        dtype = tf.float64,
        initializer = y0)

# this is the boundary of the cortical map
yb = y[:map_h]

#### main cost
yx_diff = tf.expand_dims(y, 1) - tf.expand_dims(x, 0)
yx_normsq = tf.einsum('ijk,ijk->ij', yx_diff, yx_diff)
yx_gauss = tf.exp(-1.0 * yx_normsq / (2.0 * kappa * kappa))
yx_cost = -1.0 * kappa * tf.reduce_sum(
            tf.log(
                tf.reduce_sum(yx_gauss, axis=0)))

#### regularization term 1 - within area
# n is a list of map_h * map_w objects. The i-th item of n is a list containing the indices of nodes neighboring the i-th node
n = neighborhood(map_h, map_w)
mask = tf.constant(make_mask(map_h, map_w, n), dtype=tf.float64, name='mask')

# pairwise distance: first use broadcast to calculate pairwise difference
yy_diff = tf.expand_dims(y, 1) - tf.expand_dims(y, 0)
yy_normsq = tf.einsum('ijk,ijk->ij', yy_diff, yy_diff)
yy_normsq_masked = tf.multiply(mask, yy_normsq)

reg1 = tf.reduce_sum(yy_normsq_masked)

#### regularization term 2 - boundary
# pairwise distance: first use broadcast to calculate pairwise difference
yb_diff = tf.expand_dims(yb, 1) - tf.expand_dims(b, 0)
yb_normsq = tf.einsum('ijk,ijk->ij', yb_diff, yb_diff)
bmask = make_boundary_mask(map_h)
yb_normsq_masked = tf.multiply(bmask, yb_normsq)

reg2 = tf.reduce_sum(yb_normsq_masked)

#########################
# optimization
#########################
global_step = tf.Variable(0, trainable=False)
current_eta = tf.train.exponential_decay(eta0, global_step, 10, 0.99, staircase=False)

cost = yx_cost + beta1 * reg1 + beta2 * reg2

opt = tf.train.RMSPropOptimizer(current_eta, momentum = m).minimize(cost, global_step = global_step)


def train(sess, opt, kappa, beta1, beta2, y, b1, b2, out_dir):
    sess.run(tf.global_variables_initializer())
    k = 30.0
    for i in range(1000):
        sess.run(opt, {kappa: k, beta1: b1, beta2: b2})
        k = k - k*0.005

    zz = sess.run(y)
    np.savetxt(out_dir + "y-" + "%6.5f"%b1 + '-' + "%6.5f"%b2 + ".data", zz)

###########################
# let's do it
###########################


out_dir = out_root_dir + 'sweep_' + str(sweep_id).zfill(2) + "/"
print("creating ", out_dir)
os.mkdir(out_dir)

sess = tf.Session()

for i1 in range(0, 1):
    for i2 in range(0, 1):
        b1 = 0.001*1.6**i1
        b2 = 0.001*1.6**i2
        print(b1, b2)
        train(sess, opt, kappa, beta1, beta2, y, b1, b2, out_dir)
