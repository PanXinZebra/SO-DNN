# -*- coding: utf-8 -*-

'''
This script sets the parameters that need to be specified during the running of the SO-DNN
'''

'''Algorithm main path'''
filepath="I:\\paper2021-github\\";

'''image file name'''
imagefile=filepath+"img.tif";

'''ground truth file name'''
gtfile=filepath+"gt.tif";

'''segmentated image'''
segmentimgfile=filepath+"seg.bmp";

'''ground truth colors in ISPRS dataset
Category from 0 to 5  is
0-low vegetation (LV), 
1-trees (T), 
2-buildings (B), 
3-impervious surfaces (IS), 
4-cars (C)
5-clutter/background (CB)
'''
colors=[(0,255,255),\
        (0,255,0),\
        (0,0,255),\
        (255,255,255),\
        (255,255,0),\
        (255,0,0)];
        
        
'''
category number
Vaihingen has 5 category
Potsdam has 6 category
'''        
decisionnumber=5;


'''SLIC parameter'''
thesegments=4000;
thecompactness=10;

'''segment utl class save file path'''
seghfile=filepath+"seghfile.dat";


'''input scale can only use 16,32,64,96,128,256'''
imagepatchwidth=96;

'''Generated sample by IPSSC'''
IPSSCFile=filepath+"patchsampleIPSSC.pk"

'''Generated sample by IISM'''
IISMFile=filepath+"patchsampleIISM.pk"



'''Classfy result image'''
resultimage=filepath+"resultimg.jpg";

aef=0.5;



