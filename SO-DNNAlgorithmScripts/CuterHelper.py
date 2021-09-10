# -*- coding: utf-8 -*-
import numpy as np;

def CutFromImage(image,suju):
    
    starth=suju[1];
    startl=suju[2];
    width=suju[3];
    
    hss,lss,bds=image.shape;
    
    myblock=np.zeros((width,width,bds),dtype='float32');
    
    
    inimage=1;
    
    if (starth<0 or (starth+width)>hss):
        inimage=0;
    
    if (startl<0 or (startl+width)>lss):
        inimage=0;
        
    if (inimage==1):
        myblock[:,:,:]=image[starth:starth+width,startl:startl+width,:];
    else:
        
        for ii in range(width):
            for jj in range(width):
                hpos=starth+ii;
                lpos=startl+jj;
                if (hpos>=hss):
                    hpos=hss+(hss-hpos)-1;
                else:
                    if (hpos<0):
                        hpos=-hpos;
                        
                if (lpos>=lss):
                    lpos=lss+(lss-lpos)-1;
                else:
                    if (lpos<0):
                        lpos=-lpos;
                myblock[ii,jj,:]=image[hpos,lpos,:];
   
    return myblock;


def CutFromImageSingle(image,suju):
    
    starth=suju[1];
    startl=suju[2];
    width=suju[3];
    
    hss,lss=image.shape;
    
    myblock=np.zeros((width,width),dtype='float32')-1;
    
    
    inimage=1;
    
    if (starth<0 or (starth+width)>hss):
        inimage=0;
    
    if (startl<0 or (startl+width)>lss):
        inimage=0;
        
    if (inimage==1):
        myblock[:,:]=image[starth:starth+width,startl:startl+width];
    else:
        
        for ii in range(width):
            for jj in range(width):
                hpos=starth+ii;
                lpos=startl+jj;
                if (hpos>=hss):
                    hpos=hss+(hss-hpos)-1;
                else:
                    if (hpos<0):
                        hpos=-hpos;
                        
                if (lpos>=lss):
                    lpos=lss+(lss-lpos)-1;
                else:
                    if (lpos<0):
                        lpos=-lpos;
                myblock[ii,jj]=image[hpos,lpos];
   
    return myblock;