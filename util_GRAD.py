
# coding: utf-8
import tensorflow as tf
import os
import numpy as np
slim = tf.contrib.slim
# In[1]:

def confimap_grad(confimap, config):
    """ structure loss """
    try:
        size = confimap.get_shape().as_list()
    except:
        size = confimap.shape
    b = config.batch_size
    h = size[1]
    w = size[2]
    n = h*w # size of image pixels    
    
    prewitt_x = tf.constant([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], tf.float32) # shape : (3, 3)
    prewitt_y = tf.constant([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], tf.float32) # shape : (3, 3)
    x_filter = tf.reshape(prewitt_x, [3, 3, 1, 1])  # tensorflow filtershape is (filter_size, filter_size, input, output)
    y_filter = tf.reshape(prewitt_y, [3, 3, 1, 1])
    #y_filter = tf.transpose(x_filter, [1, 0, 2, 3])

    def cal_grad_x(img):
        return tf.nn.conv2d(img, x_filter, strides=[1, 1, 1, 1], padding='SAME')
    def cal_grad_y(img):
        return tf.nn.conv2d(img, y_filter, strides=[1, 1, 1, 1], padding='SAME')
        
    return tf.reduce_mean( tf.square(cal_grad_x(confimap)) + tf.square(cal_grad_y(confimap)), axis=[1, 2] )

def confimap_grad_Temp(confimap, config):
    """ structure loss """
    try:
        size = confimap.get_shape().as_list()
    except:
        size = confimap.shape
    b = config.batch_size
    h = size[1]
    w = size[2]
    n = h*w # size of image pixels    
    
    prewitt_x =  tf.constant([[1,-1]], tf.float32)
    prewitt_y = tf.constant([[1],[-1]], tf.float32)
    x_filter = tf.reshape(prewitt_x, [1, 2, 1, 1])  # tensorflow filtershape is (filter_size, filter_size, input, output)
    y_filter = tf.reshape(prewitt_y, [2, 1, 1, 1])

    def cal_grad_x(img):
        return tf.nn.conv2d(img, x_filter, strides=[1, 1, 1, 1], padding='SAME')
    def cal_grad_y(img):
        return tf.nn.conv2d(img, y_filter, strides=[1, 1, 1, 1], padding='SAME')
        
    return tf.reduce_mean(tf.reduce_mean( tf.sqrt(tf.square(cal_grad_x(confimap)) + tf.square(cal_grad_y(confimap))), axis=[1, 2] ))
