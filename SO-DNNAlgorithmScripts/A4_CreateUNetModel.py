# -*- coding: utf-8 -*-

'''Create U-Net Model'''

from GetModel import GetModel;
import pickle;
import numpy as np; 
from keras import backend as K
from CuterHelper import *;

'''
this unet need VGG16 pretraind weights file "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
this file can search in google and download, and copy this file to a certain dir. e.g. "userhome\.keras\models"
in "CreateModelUnet.py", line 11 and 61, this file path is specified and loaded.
if do not need pretraind weights, can comment out these two lines
'''


blockwidth=imagepatchwidth;
mclasses=decisionnumber;
model=GetModel('UNET',blockwidth,blockwidth,mclasses);
model.summary();