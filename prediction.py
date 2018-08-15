# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 10:22:39 2018

@author: todd
"""

import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2

slim = tf.contrib.slim

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from notebooks import visualization

import sys
sys.path.append('../')

from SSD.nets import ssd_vgg_300, ssd_common, np_methods
from SSD.preprocessing import ssd_vgg_preprocessing


gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
sess = tf.Session(config=config)

net_shape = (300, 300)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_300.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)
# Restore SSD model.
#ckpt_filename = 'D:/presto/Python/Lib/site-packages/SSD/checkpoints/ssd_300_vgg.ckpt'
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, 'D:/presto/Python/Lib/site-packages/SSD/logs/ssd_300_vgg.ckpt')
# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)

def process_image(img, select_threshold=0.1, nms_threshold=.45, net_shape=(300, 300)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = sess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})
    
    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)
    
    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes

path = 'D:/presto/Python/Lib/site-packages/SSD/demo/'
image_names = sorted(os.listdir(path))

img = mpimg.imread(path + image_names[-1])
rclasses, rscores, rbboxes =  process_image(img)

#visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
visualization.plt_bboxes(img, rclasses, rscores, rbboxes)


#img = mpimg.imread(path + image_names[-1])
#rect = (700, 700, 375, 232)
#mask = np.zeros(img.shape[:2], np.uint8)
#bgdModel = np.zeros((1,65),np.float64)
#fgdModel = np.zeros((1,65),np.float64)
#cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
#mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
#img = img*mask2[:,:,np.newaxis]#使用蒙板来获取前景区域
#cv2.imshow('p',img)
#cv2.waitKey(0)
