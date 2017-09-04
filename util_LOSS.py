
# coding: utf-8
import tensorflow as tf
import os
import numpy as np
slim = tf.contrib.slim
# In[1]:

def strLoss(out, dpt_resize, config):
    """ structure loss """
    try:
        size = dpt_resize.get_shape().as_list()
    except:
        size = dpt_resize.shape
    b = config.batch_size
    h = size[1]
    w = size[2]
    n = h*w # size of image pixels    
    d = out - dpt_resize
    
    prewitt_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], tf.float32) # shape : (3, 3)
    x_filter = tf.reshape(prewitt_x, [3, 3, 1, 1])  # tensorflow filtershape is (filter_size, filter_size, input, output)
    y_filter = tf.transpose(x_filter, [1, 0, 2, 3])

    def cal_grad_x(img):
        return tf.nn.conv2d(img, x_filter, strides=[1, 1, 1, 1], padding='VALID')
    def cal_grad_y(img):
        return tf.nn.conv2d(img, y_filter, strides=[1, 1, 1, 1], padding='VALID')
        
    return tf.reduce_mean(tf.reduce_mean( tf.square(cal_grad_x(out - dpt_resize)) + tf.square(cal_grad_y(out - dpt_resize)), axis=[1, 2] ))

def CCC_img(x1, x2, cf):
    """img-wise"""
    try:
        size = x1.get_shape().as_list()
    except:
        size = x1.shape
    b = cf.batch_size
    h = size[1]
    w = size[2]
    n=h*w
    
    m1 = tf.reduce_mean(x1, axis=(1, 2))  # shape : b
    m2 = tf.reduce_mean(x2, axis=(1, 2))
    temp1 = tf.reshape(x1, (b, n))
    temp2 = tf.reshape(x2, (b, n))
    s1_square = tf.reduce_mean(tf.square(temp1 - m1), axis=1) # shape: b
    s2_square = tf.reduce_mean(tf.square(temp2 - m2), axis=1)
    
    s12 = tf.reduce_mean( (temp1 - m1) * (temp2 - m2), axis=1) # shape: b
    ccc = 2*s12 / (s1_square + s2_square + tf.square(m1 - m2)) # shape : b
    ccc_loss = tf.reduce_mean(1.- ccc)
    return ccc_loss

def L2(infer, gt, config):
    try:
        size = confi_map.get_shape().as_list()
    except:
        size = confi_map.shape
    b = config.batch_size
    h = size[1]
    w = size[2]
    n = h*w # size of image pixels   
    
    L2map = tf.square(infer - gt)
    L1map = tf.abs(infer - gt)
    
    return tf.reduce_mean(tf.reduce_mean(L2map, axis=[1, 2]))

def scaleInvLoss(out, dpt_resize, config):
    """ structure loss """
    try:
        size = dpt_resize.get_shape().as_list()
    except:
        size = dpt_resize.shape
    b = config.batch_size
    h = size[1]
    w = size[2]
    n = h*w # size of image pixels 
    
    return - tf.reduce_mean( tf.square( tf.reduce_sum(out - dpt_resize, axis=(1, 2)) ) / (2*n*n) )


def confiLoss(infer, gt, confi_map, var, config):
    try:
        size = confi_map.get_shape().as_list()
    except:
        size = confi_map.shape
    b = config.batch_size
    h = size[1]
    w = size[2]
    n = h*w # size of image pixels   
    
    L2map = tf.square(infer - gt)
    L1map = tf.abs(infer - gt)
    
    return tf.reduce_mean( tf.reduce_mean(L2map * confi_map, axis=[1, 2]) + (var) )

def confiStrLoss(infer, gt, confi_map, var, config):
    """ confi structure loss """
    try:
        size = confi_map.get_shape().as_list()
    except:
        size = confi_map.shape
    b = config.batch_size
    h = size[1]
    w = size[2]
    n = h*w # size of image pixels    
    
    prewitt_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], tf.float32) # shape : (3, 3)
    x_filter = tf.reshape(prewitt_x, [3, 3, 1, 1])  # tensorflow filtershape is (filter_size, filter_size, input, output)
    y_filter = tf.transpose(x_filter, [1, 0, 2, 3])

    def cal_grad_x(img):
        return tf.nn.conv2d(img, x_filter, strides=[1, 1, 1, 1], padding='VALID')
    def cal_grad_y(img):
        return tf.nn.conv2d(img, y_filter, strides=[1, 1, 1, 1], padding='VALID')
    
    grad = tf.square(cal_grad_x(infer - gt)) + tf.square(cal_grad_y(infer - gt))
    
    return tf.reduce_mean(tf.reduce_mean( grad  * confi_map, axis=[1, 2]) + (var) )

def confiLoss_grad(infer, gt, confi_map, var, confi_grad, config):
    try:
        size = confi_map.get_shape().as_list()
    except:
        size = confi_map.shape
    b = config.batch_size
    h = size[1]
    w = size[2]
    n = h*w # size of image pixels   
    
    L2map = tf.square(infer - gt)
    L1map = tf.abs(infer - gt)
    #return tf.sqrt( tf.reduce_sum(L2map * confi_map) / (n*b) )
    return tf.reduce_mean( tf.reduce_mean(L2map * confi_map, axis=[1, 2]) / (confi_grad) + (var) )

def confiStrLoss_grad(infer, gt, confi_map, var, confi_grad, config):
    """ confi structure loss """
    try:
        size = confi_map.get_shape().as_list()
    except:
        size = confi_map.shape
    b = config.batch_size
    h = size[1]
    w = size[2]
    n = h*w # size of image pixels    
    
    prewitt_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], tf.float32) # shape : (3, 3)
    x_filter = tf.reshape(prewitt_x, [3, 3, 1, 1])  # tensorflow filtershape is (filter_size, filter_size, input, output)
    y_filter = tf.transpose(x_filter, [1, 0, 2, 3])

    def cal_grad_x(img):
        return tf.nn.conv2d(img, x_filter, strides=[1, 1, 1, 1], padding='VALID')
    def cal_grad_y(img):
        return tf.nn.conv2d(img, y_filter, strides=[1, 1, 1, 1], padding='VALID')
    
    grad = tf.square(cal_grad_x(infer - gt)) + tf.square(cal_grad_y(infer - gt))
    
    return tf.reduce_mean(tf.reduce_mean( grad  * confi_map, axis=[1, 2]) /  (confi_grad) + (var) )
