# -*- coding: utf-8 -*-

from CreateModelSegnet import *;
from skimage.io import *;
import pickle;

from CreateModelFCN import *;
from CreateModelUnet import *;
from CreateModelSegnet import *;


def  GetModel(modelname,input_height,input_width,nClasses):
    model=0;
    if (modelname=='FCN8'):
        pass;
        model = PX_FCN8(nClasses=nClasses,  
             input_height = input_height, 
             input_width  = input_width);
    
    if (modelname=='FCN16'):
        pass;
        model = PX_FCN16(nClasses=nClasses,  
             input_height = input_height, 
             input_width  = input_width);
                         
    if (modelname=='FCN32'):
        pass;
        model = PX_FCN16(nClasses=nClasses,  
             input_height = input_height, 
             input_width  = input_width);
    
    if (modelname=='SEGNET'):
        pass;
        model = PX_SEGNET(nClasses=nClasses,  
             input_height = input_height, 
             input_width  = input_width);
    
    if (modelname=='UNET'):
        pass;
        model = PX_UNET(nClasses=nClasses,  
             input_height = input_height, 
             input_width  = input_width);
    
    model.compile(loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"])
    '''SEGNET cannot use adadelta need other optimizer
    sgd=keras.optimizers.SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False);
    model.compile(optimizer=sgd, loss='mean_absolute_error',metrics=['accuracy']);
    ''' 
    
    return model;


