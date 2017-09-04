
# coding: utf-8
import tensorflow as tf
import os
import numpy as np
slim = tf.contrib.slim
# In[1]:


def LogDepth(depth):
    depth = tf.maximum(depth, 1.0 / 255.0)
    return 0.179581 * tf.log(depth) + 1

def AbsRel(output, gt):
    """Absolute Relative Difference"""
    gt = tf.maximum(gt, 1.0 / 255.0) # for zero divided error
    diff = tf.reduce_mean(tf.abs(output - gt) / gt)
    return diff

def SqrRel(output, gt):
    """Squared Relative Difference"""
    gt = tf.maximum(gt, 1.0 / 255.0)
    d = output - gt
    diff = tf.reduce_mean((d * d) / gt)
    return diff

def RMSE(output, gt):
    """Root mean squared error"""
    diff = tf.sqrt(tf.reduce_mean(tf.square(output-gt)))
    return diff

def RMSE_log(output, gt):
    """Root mean squared error log-space"""
    d = LogDepth(output / 10.0) * 10.0 - LogDepth(gt / 10.0) * 10.0
    diff = tf.sqrt(tf.reduce_mean(d * d))
    return diff
    
def ScaleInvariantMSE(output, gt, is_logSpace=False):
    """Scale Invariant Mean Squared Error in (log or lin - space)"""
    if is_logSpace is True:
        ouput = LogDepth(output / 10.0) * 10.0
        gt = LogDepth(gt / 10.0) * 10.0
    d = output - gt
    diff = tf.reduce_mean(d * d)
    
    relDiff = (tf.reduce_sum(diff) * tf.reduce_sum(diff)) / tf.to_float((tf.size(d) * tf.size(d)))
    return diff - relDiff

def L2_img(output, gt):
    return tf.reduce_mean(tf.reduce_mean(tf.square(output - gt), axis=[1, 2]))

def L1_img(output, gt):
    return tf.reduce_mean(tf.reduce_mean(tf.abs(output - gt), axis=[1, 2]))

def RMSE_img(output, gt):
    return tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.square(output - gt), axis=[1, 2])))

def AbsRel_img(output, gt):
    return tf.reduce_mean(tf.reduce_mean(tf.abs(output - gt) / gt, axis=[1, 2]))

def SqrRel_img(output, gt):
    return tf.reduce_mean(tf.reduce_mean(tf.square(output-gt) / gt, axis=[1, 2]))


def Log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def Log10Error(output, gt):
    output = tf.maximum(output, 1.0 / 255.0)
    gt = tf.maximum(gt, 1.0 / 255.0)
    diff = tf.mean(tf.abs(Log10(output) - Log10(gt)))
    return diff

def Threshold(output, gt, threshold):
    output = tf.maximum(output, 1.0 / 255.0)
    gt = tf.maximum(gt, 1.0 / 255.0)
    size=tf.size(tf.where(tf.maximum(output / gt, gt / output) < threshold)[0])
    #withinThresholdCount = tf.size(tf.to_int32(tf.where(tf.maximum(output / gt, gt / output) < threshold )[0]))
    withinThresholdCount = tf.to_float(size)
    return withinThresholdCount / tf.to_float(tf.size(gt))

