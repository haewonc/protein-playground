import numpy as np
import scipy.spatial
import string
import tensorflow as tf

# read A3M and convert letters into
# integers in the 0..20 range
def parse_a3m(filename):
    seqs = []
    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))
    
    # read file line by line
    for line in open(filename,"r"):
        # skip labels
        if line[0] != '>':
            # remove lowercase letters and right whitespaces
            seqs.append(line.rstrip().translate(table))

    # convert letters into numbers
    alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-"), dtype='|S1').view(np.uint8)
    msa = np.array([list(s) for s in seqs], dtype='|S1').view(np.uint8)
    for i in range(alphabet.shape[0]):
        msa[msa == alphabet[i]] = i

    # treat all unknown characters as gaps
    msa[msa > 20] = 20

    return msa

# characters to integers
def aa2idx(seq):

    # convert letters into numbers
    abc = np.array(list("ARNDCQEGHILKMFPSTWYV-"), dtype='|S1').view(np.uint8)
    idx = np.array(list(seq), dtype='|S1').view(np.uint8)
    for i in range(abc.shape[0]):
        idx[idx == abc[i]] = i

    # treat all unknown characters as gaps
    idx[idx > 20] = 20

    return idx

# integers back to sequence:
def idx2aa(idx):
    abc=np.array(list("ARNDCQEGHILKMFPSTWYV"))
    return("".join(list(abc[idx])))

# 1-hot MSA to PSSM
def msa2pssm(msa1hot, w):
    beff = tf.reduce_sum(w)
    f_i = tf.reduce_sum(w[:,None,None]*msa1hot, axis=0) / beff + 1e-9
    h_i = tf.reduce_sum( -f_i * tf.log(f_i), axis=1)    
    return tf.concat([f_i, h_i[:,None]], axis=1)

# reweight MSA based on cutoff
def reweight(msa1hot, cutoff):
    with tf.name_scope('reweight'):
        id_min = tf.cast(tf.shape(msa1hot)[1], tf.float32) * cutoff
        id_mtx = tf.tensordot(msa1hot, msa1hot, [[1,2], [1,2]])
        id_mask = id_mtx > id_min
        w = 1.0/tf.reduce_sum(tf.cast(id_mask, dtype=tf.float32),-1)
    return w

def fast_dca(msa1hot, weights, penalty = 4.5):
    
    nr = tf.shape(msa1hot)[0]
    nc = tf.shape(msa1hot)[1]
    ns = tf.shape(msa1hot)[2]

    with tf.name_scope('covariance'):
        x = tf.reshape(msa1hot, (nr, nc * ns))
        num_points = tf.reduce_sum(weights) - tf.sqrt(tf.reduce_mean(weights))
        mean = tf.reduce_sum(x * weights[:,None], axis=0, keepdims=True) / num_points
        x = (x - mean) * tf.sqrt(weights[:,None])
        cov = tf.matmul(tf.transpose(x), x)/num_points

    with tf.name_scope('inv_convariance'):
        cov_reg = cov + tf.eye(nc * ns) * penalty / tf.sqrt(tf.reduce_sum(weights))
        inv_cov = tf.linalg.inv(cov_reg)

        x1 = tf.reshape(inv_cov,(nc, ns, nc, ns))
        x2 = tf.transpose(x1, [0,2,1,3])
        features = tf.reshape(x2, (nc, nc, ns * ns))

        x3 = tf.sqrt(tf.reduce_sum(tf.square(x1[:,:-1,:,:-1]),(1,3))) * (1-tf.eye(nc))
        apc = tf.reduce_sum(x3,0,keepdims=True) * tf.reduce_sum(x3,1,keepdims=True) / tf.reduce_sum(x3)
        contacts = (x3 - apc) * (1-tf.eye(nc))

    return tf.concat([features, contacts[:,:,None]], axis=2)


def get_dihedrals(a, b, c, d):

    b0 = -1.0*(b - a)
    b1 = c - b
    b2 = d - c

    b1 /= np.linalg.norm(b1, axis=-1)[:,None]

    v = b0 - np.sum(b0*b1, axis=-1)[:,None]*b1
    w = b2 - np.sum(b2*b1, axis=-1)[:,None]*b1

    x = np.sum(v*w, axis=-1)
    y = np.sum(np.cross(b1, v)*w, axis=-1)

    return np.arctan2(y, x)


def get_angles(a, b, c):

    v = a - b
    v /= np.linalg.norm(v, axis=-1)[:,None]

    w = c - b
    w /= np.linalg.norm(w, axis=-1)[:,None]

    x = np.sum(v*w, axis=1)

    return np.arccos(x)

def seq_to_in(self, seq):
    ncol = tf.shape(seq)[1]
    msa1hot  = tf.one_hot(seq, 21)
    weight = reweight(msa1hot, 0.8)
    f1d_seq = msa1hot[0,:,:20]
    f1d_pssm = msa2pssm(msa1hot, weight)
    f1d = tf.concat(values=[f1d_seq, f1d_pssm], axis=1)
    f1d = tf.expand_dims(f1d, axis=0)
    f1d = tf.reshape(f1d, [1,ncol,42])
    f2d_dca = tf.zeros([ncol,ncol,442], tf.float32)
    f2d_dca = tf.expand_dims(f2d_dca, axis=0)
    f2d = tf.concat([tf.tile(f1d[:,:,None,:], [1,1,ncol,1]),
                    tf.tile(f1d[:,None,:,:], [1,ncol,1,1]),
                    f2d_dca], axis=-1)
    f2d = tf.reshape(f2d, [1,ncol,ncol,442+2*42])
    return f2d