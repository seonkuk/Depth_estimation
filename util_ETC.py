
# coding: utf-8
import tensorflow as tf
import os
slim = tf.contrib.slim
# In[1]:

def save(path, name, saver, sess, count):
    if not os.path.exists(path):
        os.makedirs(path)
    saver.save(sess, os.path.join(path, name), global_step = count)


def load(path, saver, sess):
    ckpt = tf.train.get_checkpoint_state(path)
    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    saver.restore(sess, os.path.join(path, ckpt_name))
    #print ("Successfully loaded:,", os.path.join(path, ckpt_name))


def ckpt_check_and_load(path, saver ,sess, is_train=False):
    if is_train: 
        print ("check_ckpt for test during training.")        
    ckpt = tf.train.get_checkpoint_state(path)
    if ckpt and ckpt.model_checkpoint_path:
        try:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Successfully loaded:", ckpt.model_checkpoint_path)
        except:
            print("Error on loading old network weights")
    else:
        print("Could not find old network weights")  

def bring_BN_movingAVG_include_prefix(prefix=None):
    moving_avgs=[]
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    for i in update_ops:
        if i.name.startswith(prefix):
            moving_avgs.append(i)
    return moving_avgs

def bring_var_except_prefix(prefix=None):
    train_var=[]
    for i in tf.global_variables():
        if not i.name.startswith(prefix):
            train_var.append(i)
    return train_var

def bring_var_include_prefix(prefix=None):
    train_var=[]
    for i in tf.global_variables():
        if i.name.startswith(prefix):
            train_var.append(i)
    return train_var
