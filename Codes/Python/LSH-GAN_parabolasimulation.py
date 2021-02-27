import tensorflow as tf
import numpy as np
#from training_data import *
#import seaborn as sb
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from scipy import io, sparse
from numpy import genfromtxt
from multiprocessing import Pool
from sklearn.neighbors import LSHForest

tf.disable_v2_behavior() 
#sb.set()

def get_y(x):
    return 10 + x*x


def sample_data(n=10000, scale=100):
    data = []

    x = scale*(np.random.random_sample((n,))-0.5)

    for i in range(n):
        yi = get_y(x[i])
        data.append([x[i], yi])

    return np.array(data)
def chunk(a, n):
    k, m = divmod(len(a), n)
    return list(tuple(a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]) for i in range(n))


# Define the main LSH Sampling
def lsh_main(data,obs):
    global lshf 
    lshf = LSHForest(n_estimators=10, random_state=42)
    lshf.fit(sparse.coo_matrix(data)) 
    query_sets = chunk(range(obs),p)
    pool = Pool(processes=p)  
    NN_set = pool.map(knn, query_sets)
    pool.close()
    indices = np.vstack(NN_set)
    arr1=np.ones(obs)
    Nb=np.zeros(4)
    m=np.zeros(4)
    for i in range(0,obs):
        if arr1[i]!=0:
            Nb = indices[i][1:5]
            arr1[Nb]=m
    return arr1

# Define the Nearesr Neighbor search with trained LSH data
def knn(q_idx):
    distances, indices = lshf.kneighbors(Xnew[q_idx,:], n_neighbors=5)
    return indices



def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

def generator(Z,hsize=[16, 16],reuse=False):
    with tf.variable_scope("GAN/Generator",reuse=reuse):
        h1 = tf.layers.dense(Z,hsize[0],activation=tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1,hsize[1],activation=tf.nn.leaky_relu)
        out = tf.layers.dense(h2,2)

    return out

def discriminator(X,hsize=[16, 16],reuse=False):
    with tf.variable_scope("GAN/Discriminator",reuse=reuse):
        h1 = tf.layers.dense(X,hsize[0],activation=tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1,hsize[1],activation=tf.nn.leaky_relu)
        h3 = tf.layers.dense(h2,2)
        out = tf.layers.dense(h3,1)

    return out, h3


X = tf.placeholder(tf.float32,[None,2])
Z = tf.placeholder(tf.float32,[None,2])

G_sample = generator(Z)
r_logits, r_rep = discriminator(X)
f_logits, g_rep = discriminator(G_sample,reuse=True)

disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits,labels=tf.ones_like(r_logits)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits,labels=tf.zeros_like(f_logits)))
gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits,labels=tf.ones_like(f_logits)))

gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Generator")
disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Discriminator")

gen_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(gen_loss,var_list = gen_vars) # G Train step
disc_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(disc_loss,var_list = disc_vars) # D Train step



# sess = tf.Session(config=config)
sess = tf.Session()
tf.global_variables_initializer().run(session=sess)

batch_size = 256
nd_steps = 10
ng_steps = 10

x_plot = sample_data(n=batch_size)
Xnew=x_plot
p=20
row=x_plot.shape[0] 
col=x_plot.shape[1]

for i in range(0,1):
    row=Xnew.shape[0]
    result=lsh_main(Xnew,row)
    c=np.nonzero(result)
    c1=c[0]
    Xnew=Xnew[c1,:]


for i in range(10001):
    X_batch = sample_data(n=batch_size)
    #Z_batch = sample_Z(batch_size, 2)
    row1=row-Xnew.shape[0]
    da1= sample_Z(row1,col)
    Z_batch=np.row_stack((da1,Xnew))


    for _ in range(nd_steps):
        _, dloss = sess.run([disc_step, disc_loss], feed_dict={X: X_batch, Z: Z_batch})
    rrep_dstep, grep_dstep = sess.run([r_rep, g_rep], feed_dict={X: X_batch, Z: Z_batch})

    for _ in range(ng_steps):
        _, gloss = sess.run([gen_step, gen_loss], feed_dict={Z: Z_batch})

    rrep_gstep, grep_gstep = sess.run([r_rep, g_rep], feed_dict={X: X_batch, Z: Z_batch})

    print("Iterations: %d\t Discriminator loss: %.4f\t Generator loss: %.4f"%(i,dloss,gloss))
    if i%1000 == 0:
        plt.figure()
        g_plot = sess.run(G_sample, feed_dict={Z: Z_batch})
        xax = plt.scatter(x_plot[:,0], x_plot[:,1],s=1)
        gax = plt.scatter(g_plot[:,0],g_plot[:,1],s=1)

        plt.legend((xax,gax), ("Real Data","Generated Data"))
        plt.title('Samples at Iteration %d'%i)
        plt.tight_layout()
        plt.savefig('iteration_new_%d.png'%i)
        plt.close()

