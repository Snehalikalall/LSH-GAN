#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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




import timeit

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
data = genfromtxt('preprocessdata.csv',delimiter=",") #Give the data here
x_plot = data
Xnew=x_plot
p=20
row=x_plot.shape[0] 
col=x_plot.shape[1]

#Number of iteration itr=1 for all except klein. itr=2 for klein
itr=1

for i in range(0,itr):
    rowlsh=Xnew.shape[0]
    result=lsh_main(Xnew,rowlsh)
    c=np.nonzero(result)
    c1=c[0]
    Xnew=Xnew[c1,:]



def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

def generator(Z,hsize=[16, 16],reuse=False):
    with tf.variable_scope("GAN/Generator",reuse=reuse):
        h1 = tf.layers.dense(Z,hsize[0],activation=tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1,hsize[1],activation=tf.nn.leaky_relu)
        out = tf.layers.dense(h2,col)

    return out

def discriminator(X,hsize=[16, 16],reuse=False):
    with tf.variable_scope("GAN/Discriminator",reuse=reuse):
        h1 = tf.layers.dense(X,hsize[0],activation=tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1,hsize[1],activation=tf.nn.leaky_relu)
        h3 = tf.layers.dense(h2,col)
        out = tf.layers.dense(h3,1)

    return out, h3


X = tf.placeholder(tf.float32,[None,col])
Z = tf.placeholder(tf.float32,[None,col])

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

nd_steps = 10
ng_steps = 10




start = timeit.default_timer()
#Epoch size.................

for i in range(10001):
    X_batch = x_plot
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
    #plt.figure()
    #g_plot = sess.run(G_sample, feed_dict={Z: Z_batch})
    #xax = plt.scatter(x_plot[:,0], x_plot[:,1],s=1)
    #gax = plt.scatter(g_plot[:,0],g_plot[:,1],s=1)
    #plt.legend((xax,gax), ("Real Data","Generated Data"))
    #plt.title('Samples at Iteration')
    #plt.tight_layout()
    #plt.savefig('iteration_trad_1.png')
    #plt.close()
        #if i%10 == 0:
    #    f.write("%d,%f,%f\n"%(i,dloss,gloss))

stop = timeit.default_timer()
print('Time: ', stop - start)




import pandas as pd
start = timeit.default_timer()
feat_size=col
#Size of Generated Data
batch_size=(np.arange(0.25,1.75,0.25)*feat_size).astype(int)
for i in range(6):
    row1=batch_size[i]-Xnew.shape[0]
    da1= sample_Z(row1,feat_size)
    Z_batch=np.row_stack((da1,Xnew))
    g_plot = sess.run(G_sample, feed_dict={Z: Z_batch})
    #Saving the generated Data
    f = open("data_mixdata_iter"+str(i)+".csv","w")
    g_plot_pd=pd.DataFrame(g_plot)
    g_plot_pd.to_csv(f,index=False,header=False)
    print(g_plot_pd.shape)
stop = timeit.default_timer()
print('Time: ', stop - start)

