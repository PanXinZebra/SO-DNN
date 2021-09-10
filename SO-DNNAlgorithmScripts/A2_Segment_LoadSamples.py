# -*- coding: utf-8 -*-

import sys
sys.path.append(".")
'''
Segment image, load point based samples, and load ground truth image.
'''
from SegHelper import *;
from PointHelper import *;

import numpy as np;
from skimage.transform import *;
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed, join_segmentations
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage.io import *;
import pickle;
import copy;
from skimage import feature
from skimage.filters import meijering, sato, frangi, hessian
import matplotlib.pyplot as plt




image=imread(imagefile);

nmimage=image/255;

gtimage=imread(gtfile);

segments=slic(image,n_segments=thesegments,compactness=thecompactness);

boundarymimg=mark_boundaries(image,segments);

imsave(segmentimgfile,boundarymimg);

segh=SegHelper();

segh.readSegments(segments);

segh.savefile(seghfile);

'''
Load point based-samples  01.ptp to 05.ptp; 01 corresponding to low vegetation (LV), and so on
point based-samples are create by tools "SampleSelecter".
"SampleSelecter" is a c# project in Tools directory, "
'''
trainrois=[];
for ii in range(decisionnumber):
    ptpfilename=filepath+"%02d"%(ii+1)+".ptp";
    print(ptpfilename);
    dicsion=ii;
    trainrois.append(ptpfilename);
    trainrois.append(dicsion);
pointsarray=DFPointsArray(trainrois);       
pointsarray.toString(); 