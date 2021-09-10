# -*- coding: utf-8 -*-

from keras.utils.data_utils import get_file
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras, sys, time, warnings
from keras.models import *
from keras.layers import *
from keras.layers.core import Activation, Reshape

VGG_Weights_path=get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5','aa');

def PX_UNET( nClasses ,  input_height=224, input_width=224):
    ## input_height and width must be devisible by 32 because maxpooling with filter size = (2,2) is operated 5 times,
    ## which makes the input_height and width 2^5 = 32 times smaller
    #assert input_height%32 == 0
    #assert input_width%32 == 0
    IMAGE_ORDERING =  "channels_last" 

    img_input = Input(shape=(input_height,input_width, 3)) ## Assume 224,224,3
    
    ## Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=IMAGE_ORDERING )(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING )(x)
    pool1 = x  #112  2  8
    
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING )(x)
    pool2 = x # 56   4  4

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format=IMAGE_ORDERING )(x)
    pool3 = x # 28  8   2

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format=IMAGE_ORDERING )(x)
    if (not (input_height==8)):
        pass;
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format=IMAGE_ORDERING )(x)## (None, 14, 14, 512) 
    pool4 = x # 14  16  2

    # Block 5
    
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format=IMAGE_ORDERING )(x)
    if (not (input_height==16 or input_height==8)):
        pass;
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format=IMAGE_ORDERING )(x)## (None, 7, 7, 512)
    pool5 = x # 7  32
    
    vgg  = Model(  img_input , pool5  )
    vgg.load_weights(VGG_Weights_path) ## loading VGG weights for the encoder parts of FCN8
    
    
    
    if (not (input_height==16 or input_height==8)):
        pass;
        x = Conv2D(512, (3, 3), activation="relu",name='Upblock_1_cov1', padding="same")(x)
        x = Conv2D(512, (3, 3), activation="relu",name='Upblock_1_cov2', padding="same")(x)
        x = UpSampling2D(size=(2, 2),name='upblock1_Upsample1')(x);#14
        x = concatenate([x, pool4], axis=3)
    
    if (not (input_height==8)):
        pass;
        x = Conv2D(256, (3, 3), activation="relu",name='Upblock_2_cov1', padding="same")(x)
        x = Conv2D(256, (3, 3), activation="relu",name='Upblock_2_cov2', padding="same")(x)
        x = UpSampling2D(size=(2, 2),name='upblock2_Upsample1')(x);#28
        x = concatenate([x, pool3], axis=3)

   
    x = Conv2D(128, (3, 3), activation="relu",name='Upblock_3_cov1', padding="same")(x)
    x = Conv2D(128, (3, 3), activation="relu",name='Upblock_3_cov2', padding="same")(x)
    x = UpSampling2D(size=(2, 2),name='upblock3_Upsample1')(x);#56
    x = concatenate([x, pool2], axis=3)
    
    x = Conv2D(64, (3, 3), activation="relu",name='Upblock_4_cov1', padding="same")(x)
    x = Conv2D(64, (3, 3), activation="relu",name='Upblock_4_cov2', padding="same")(x)
    x = UpSampling2D(size=(2, 2),name='upblock4_Upsample1')(x);#112
    x = concatenate([x, pool1], axis=3)
    
    x = Conv2D(32, (3, 3), activation="relu",name='Upblock_5_cov1', padding="same")(x)
    x = Conv2D(32, (3, 3), activation="relu",name='Upblock_5_cov2', padding="same")(x)
    x = UpSampling2D(size=(2, 2),name='upblock5_Upsample1')(x);#112
    
    x = Convolution2D(nClasses, (1, 1), padding="valid")(x)
    
    outputs = Activation("softmax")(x);
     
    model = Model(img_input, outputs)

    return model