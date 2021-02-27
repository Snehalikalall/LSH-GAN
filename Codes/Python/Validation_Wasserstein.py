#!/usr/bin/env python
# coding: utf-8

# In[50]:


import ot 
import numpy as np
from scipy.spatial.distance import cdist
# Original preprocessed dataset (default yan dataset)
data = np.genfromtxt('yand.csv',delimiter=",") 
# Enlarged Dataset from GAN
resgan=np.genfromtxt('resgan.csv',delimiter=",") 
unifs1 = data / len(data)
unifs2 = resgan / len(resgan)
dist_mat = cdist(unifs1, unifs2, 'euclid')
emd_dists = ot.emd2(np.ones(len(data)) / len(data), np.ones(len(resgan)) / len(resgan), dist_mat,numItermax=100000)
print(emd_dists)

