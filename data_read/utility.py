
# coding: utf-8

# In[3]:

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
import h5py

print ("version of TF : ", tf.__version__)

def read_mat(path):
    """
    open a mat file. 
    input : path, ex) "label.mat"
    """
    try:
        import scipy.io
        return scipy.io.loadmat(path)
    except:
        import h5py
        return h5py.File(path)
