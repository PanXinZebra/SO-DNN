# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

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
from skimage.measure import shannon_entropy
from skimage.morphology import flood;
from skimage import feature



class SegHelper:
    '''           
    A imagesegment's property
    ID===ID Number;
    Pixels-Row and Pixels-Col == Pixels position list
    Position-Rect  outer frame of a rectange
    W/H  Width and Hight
    CenterPoint CenterPoint of a segment
    Decsion0,Decsion1,Decsion2 Corresponding Decsion
    
    Decsion 0 is wether segment is pure segment
    
    '''
    imgseglist:dict;
    
    '''segment image'''
    imgseg:np.array;
    
    '''segment number'''
    numsegs:int;
    
    def __init__(self):
        pass;
        
    '''Read image-segments result into this seghelper'''    
    def readSegments(self, imagesegments):
        maxvalue=np.max(imagesegments);
        minvalue=np.max(imagesegments);
        seglist=dict();
        uniquelist=np.unique(imagesegments);
        for ii in uniquelist:
            members=dict();
            members["ID"]=ii;
            members["Pixels-Row"]=[];
            members["Pixels-Col"]=[];
            members["Position-Rect"]=[];
            members["W/H"]=[];
            members["CenterPoint"]=[];
            
            
            members["Decsion0"]=0;
            members["Decsion1"]=0;
            members["Decsion2"]=0;
            
            members["NumofPixels"]=0;
            
            '''新加入的辅助决策变量'''
            members["RightDecsion"]=0;
            members["MLPDec"]=0;
            members["MLPDecFuzzy"]=0;
            ss=[8,16,32,64,96,128];
            for ess in ss:
                ascale=ess;
                members["CNNDec%02d"%ascale]=0;
                members["CNNDecFuzzy%02d"%ascale]=0;
                members["UNETDec%02d"%ascale]=0;
                members["UNETDecFuzzy%02d"%ascale]=0;
                members["UNETSINGLEDec%02d"%ascale]=0;
                members["UNETSINGLEDecFuzzy%02d"%ascale]=0;
                members["UNETVTDec%02d"%ascale]=0;
                members["UNETVTFuzzy%02d"%ascale]=0;
                
            
            seglist[ii]=members;
        
        [hss,lss]=np.shape(imagesegments);
        
        '''Set Pixels-Row and Pixels-Col ='''
        for ii in range(hss):
            for jj in range(lss):
                idpos=imagesegments[ii,jj];
                seglist[idpos]["Pixels-Row"].append(ii);
                seglist[idpos]["Pixels-Col"].append(jj);
                
        '''Set Position-Rect, W/H, CenterPoint, NumofPixels'''  
        for ii in uniquelist:
            minrow=min(seglist[ii]["Pixels-Row"]);
            maxrow=max(seglist[ii]["Pixels-Row"]);
            mincol=min(seglist[ii]["Pixels-Col"]);
            maxcol=max(seglist[ii]["Pixels-Col"]);
            seglist[ii]["Position-Rect"]=[minrow, mincol, maxrow, maxcol];
            seglist[ii]["W/H"]=[maxcol-mincol,maxrow-minrow];
            crow=int((maxrow+minrow)/2);
            ccol=int((maxcol+mincol)/2);
            seglist[ii]["CenterPoint"]=[crow, ccol];
            seglist[ii]["NumofPixels"]=len(seglist[ii]["Pixels-Row"]);
            
        self.imgseglist=seglist;
        self.numsegs=maxvalue+1;
        self.imgseg=imagesegments;
        
    
    
    def toString(self):
        
        for ii in range(self.numsegs):
            print("%03d----%d"%(ii,self.getNumofPixel(ii)));
        
    
    def getNumofPixel(self, theID:int) -> int:
        return self.imgseglist[theID]["NumofPixels"];
    
    
    def getTheIDfromRowandCol(self, row,col) -> int:
        return self.imgseg[row,col];
        
    
    def savefile(self, filename):
        kk=[self.imgseg, self.imgseglist, self.numsegs];
        wf=open(filename,'wb');
        pickle.dump(kk,wf);
        wf.close();
    
    def readfile(self, filename):
        wf=open(filename,'rb');
        [self.imgseg, self.imgseglist, self.numsegs]=pickle.load(wf);
        wf.close();  
    
    def setDecsionValue(self, key, decv, value):
        self.imgseglist[key][decv]=value;
        
    
    def saveColoredDecsion(self,filename,colors,colorname="Decsion0"):
        
        [hss,lss]=np.shape(self.imgseg);
        resultimage=np.zeros((hss,lss,3),dtype='uint8');
       
        
        allkeys=self.imgseglist.keys();
        for thekey in allkeys:
            segment=self.imgseglist[thekey];
            hh=segment["Pixels-Row"];
            ll=segment["Pixels-Col"];
            coloridx=segment[colorname];
            resultimage[hh,ll,:]=colors[coloridx];

        
        
        imsave(filename,resultimage);
        
        
        pass;
        
        
    def savePureDecsion(self,filename,puredecsioname="Decsion0"):
        pass;
        [hss,lss]=np.shape(self.imgseg);
        resultimage=np.zeros((hss,lss,3),dtype='uint8');
        color1=np.asarray((0,0,0));        
        color2=np.asarray((0,0,255));
        
        allkeys=self.imgseglist.keys();
        for thekey in allkeys:
            segment=self.imgseglist[thekey];
            hh=segment["Pixels-Row"];
            ll=segment["Pixels-Col"];
            if (segment[puredecsioname]>1):
                resultimage[hh,ll,:]=color2;
            else:
                resultimage[hh,ll,:]=color1;
        
        
        imsave(filename,resultimage);
    
    def saveAnalysisDecsion(self,filename,analysisdecsioname="Decsion1"):
        pass;
        [hss,lss]=np.shape(self.imgseg);
        resultimage=np.zeros((hss,lss),dtype='float');
        color1=np.asarray((0,0,0));        
        color2=np.asarray((0,0,255));
        
        allkeys=self.imgseglist.keys();
        for thekey in allkeys:
            segment=self.imgseglist[thekey];
            hh=segment["Pixels-Row"];
            ll=segment["Pixels-Col"];
            value=segment[analysisdecsioname];
            resultimage[hh,ll]=value;
            
        maxv=np.max(resultimage);
        minv=np.min(resultimage);
        
        resultimage=(resultimage-minv)/(maxv-minv);
        
        imsave(filename,resultimage);

    def outputDecsion(self):
        result1=[];
        result2=[];
        allkeys=self.imgseglist.keys();
        for thekey in allkeys:
            segment=self.imgseglist[thekey];
            value=segment["Decsion0"];
            value2=segment["Decsion1"];
            if (value>1):
                result2.append(value2);
            else:
                result1.append(value2);
                
        return (result1,result2);
    
    
    def markBoundaryOnCrossSegment(self,aimage):
        '''mark boundarys according Decsion0>1'''
        allkeys=self.imgseglist.keys();
        #no cross border
        corresSegment1=np.copy(self.imgseg);
        
        #corss border
        corresSegment2=np.copy(self.imgseg);
        
        for thekey in allkeys:
            segment=self.imgseglist[thekey];
            value=segment["Decsion0"];
            hh=segment["Pixels-Row"];
            ll=segment["Pixels-Col"];
            if (value>1):
                corresSegment1[hh,ll]=0;
            else:
                corresSegment2[hh,ll]=0;
                
        im1=mark_boundaries(aimage,corresSegment1,mode='inner',background_label=0)
        im2=mark_boundaries(im1,corresSegment2,color=(0,0,1),mode='inner',background_label=0)
        
        return im2;
    
    
    def transformName(self,name1,name2):
        '''将name1 转成name2'''
        pass;
        allkeys=self.imgseglist.keys();
        for thekey in allkeys:
            segment=self.imgseglist[thekey];
            segment[name2]=segment[name1];
        return;
            
        
        


        
class DecsionImageHelper:
    '''
    Input a rgb image, and translate it into a int categorylabel image
    '''
    image:np.array;
    
    
    colorlist:[];
    colornum:int;
    npcolorlist:np.array;
    
    
    
    def __init__(self):
        pass;
        self.setColorList();
        
    '''inputimage and read it into image '''
    def readImage(self, decimage):
        pass;
        
        [hss,lss,bds]=np.shape(decimage);
        
        if bds==4:
            decimage=decimage[:,:,0:3];
            '''remove transperant bands'''
            
        
        self.image=np.zeros((hss,lss),dtype='int');
        
        for ii in range(hss):
            for jj in range(lss):
                tvalue=decimage[ii,jj,:];
                self.image[ii,jj]=self.getLabel(tvalue);
                
    
        
    
    def getLabel(self, inputvalue):
        pass;
        vv=np.sum(np.abs(self.npcolorlist-inputvalue),1);
        return np.argmin(vv);
        
    
    def setColorList(self, alist=None):
        if alist is not None:
            self.colorlist=alist;
            self.colornum=len(self.colorlist);  
            self.npcolorlist=np.asarray(self.colorlist);
            return;
        '''
        Set Plate==Vaihingen and Potsdam default value
      
        '''
        self.colorlist=[(0,255,255),\
                        (0,255,0),\
                        (0,0,255),\
                        (255,255,255),\
                        (255,255,0),\
                        (255,0,0)];
        self.colornum=len(self.colorlist);  
        self.npcolorlist=np.asarray(self.colorlist);

    def clone(self):
        return copy.deepcopy(self);
    
    
    '''Transform decision based on segments'''
    def imageSegTrans(self, sh:SegHelper, puredecsioname="Decsion0"):
        allkeys=sh.imgseglist.keys();
        for thekey in allkeys:
            segment=sh.imgseglist[thekey];
            hlist=segment["Pixels-Row"];
            llist=segment["Pixels-Col"];
            items=self.image[hlist,llist];
            [unique,count]=np.unique(items,return_counts=True)
            pos=np.argmax(count);
            realvalue=int(unique[pos]);
            self.image[hlist,llist]=realvalue;
            lnum=np.shape(unique);
            segment[puredecsioname]=realvalue;
            
            
            
                        
    def getBMP(self):
        [hss,lss]=np.shape(self.image);
        resultimage=np.zeros((hss,lss,3),dtype='uint8');
        
        for ii in range(hss):
            for jj in range(lss):
                value=self.image[ii,jj];
                resultimage[ii,jj,:]=self.npcolorlist[value,:];
        return resultimage;
    
    def saveBMP(self,filename):
        img=self.getBMP();
        imsave(filename,img);

    

'''
Cut images from image, SegHelper helper    
''' 
class CutImageHelper:
    
 
    
    '''cut image'''
    cutimage:np.array;
    
    '''cut seglables'''
    cutseg:np.array;
    
    '''cut edge analysis image'''
    cutedgeimage:np.array;
    
    '''masked image, the part of image which !=cutid  will be filled with 0 '''
    cutmaskedimage:np.array;
 
    '''基于cutedgeimage 增强的cutimage rainforce by e'''
    cutenforceimage:np.array;
    
    cutenforceedgeimage:np.array;
    
    '''true false maskimage image'''
    themask:np.array;
    
    cutid:int;
    cutwidth:int;
    cutheight:int;
    
    '''sub image segment result'''
    cutreseg:np.array;
    
    '''segment result id list'''
    maskedidlist:np.array;
    
    

    
    def __init__(self):
        pass;
    
    
    def cutSubImage(self, image:np.array, seghelper:SegHelper, segid:int, width:int=0, pedgesimage:np.array=None):
        pass;
        
        self.cutid=segid;
        
        if (width==0):
            '''cut a outer frame of image'''
            [minrow, mincol, maxrow, maxcol]=seghelper.imgseglist[self.cutid]["Position-Rect"];
            self.cutimage=image[minrow:maxrow+1,mincol:maxcol+1];
            self.cutseg=seghelper.imgseg[minrow:maxrow+1,mincol:maxcol+1];
            self.cutwidth=maxcol-mincol+1;
            self.cutheight=maxrow-minrow+1;
            
            if (pedgesimage is not None):
                pass;
                self.cutedgeimage=pedgesimage[minrow:maxrow+1,mincol:maxcol+1]
            
        else:
            pass;
            '''cut image based on width '''
               
        [hh,ll]=np.where(self.cutseg!=self.cutid);
        self.cutmaskedimage=np.copy(self.cutimage);
        self.cutmaskedimage[hh,ll]=0;
        self.themask=np.ones((self.cutheight,self.cutwidth),dtype='bool');
        self.themask[hh,ll]=False;
            
    
        if (pedgesimage is not None):

            
         
            [hh,ll]=np.where(self.cutseg==self.cutid);
            imgv=self.cutimage[hh,ll];
           
            
            imgvmax=np.max(imgv);
            imgvmin=np.min(imgv);
            
            '''
            edgv=self.cutedgeimage[hh,ll];
            edgvmax=np.max(edgv);
            edgvmin=np.min(edgv);
            
            enforce=(self.cutedgeimage-edgvmin)/(edgvmax-edgvmin)
            self.cutenforceedgeimage=enforce;
            
            
            enforce=np.stack((enforce,enforce,enforce),2);
            self.cutenforceimage=(enforce)/2.0;
            '''
            
            
            self.cutenforceimage=(self.cutimage-imgvmin)/(imgvmax-imgvmin);
            tempedgv = sobel(rgb2gray(self.cutenforceimage));
            edgv=tempedgv[hh,ll];
            edgvmax=np.max(edgv);
            edgvmin=np.min(edgv);
            enforce=(tempedgv-edgvmin)/(edgvmax-edgvmin)
            self.cutenforceedgeimage=enforce;
            
    
    
    def toString(self):
        pass;
        print(np.shape(self.cutimage));
        print(np.shape(self.cutseg));
        print(np.shape(self.cutmaskedimage));
        
        
        
    def reSegment(self,inputmarkers=4, inputcompactness=0.001):
        '''Perform segment on this small image'''
        gradient = sobel(rgb2gray(self.cutmaskedimage));
        segments = watershed(gradient, markers=inputmarkers, compactness=inputcompactness,mask=self.themask);
        self.cutreseg=segments;
    
    
    def reSegment2(self, segmentsnum=6,inputcompactness=10):
        '''Perform segment on this small image'''
        segments=slic(self.cutmaskedimage,n_segments=segmentsnum,compactness=inputcompactness,mask=self.themask);
    
        self.cutreseg=segments;
    
    def reSegment3(self, segmentsnum=6,inputcompactness=10):
        '''Perform segment on this small image'''
        #segments=slic(self.cutmaskedimage,n_segments=segmentsnum,compactness=inputcompactness,mask=self.themask);
        segments=slic(self.cutenforceimage,n_segments=segmentsnum,compactness=inputcompactness,mask=self.themask,sigma=3);
        #segments=slic(self.cutedgeimage,n_segments=segmentsnum,compactness=inputcompactness,mask=self.themask);
        self.cutreseg=segments;
   
    def reSegment4(self, segmentsnum=6,inputcompactness=10):
        '''Perform segment on this small image'''
        segments=felzenszwalb(self.cutmaskedimage, scale=40, sigma=2, min_size=20)
      
        self.cutreseg=segments;
   
    def reSegment5(self, segmentsnum=6,inputcompactness=10):
        '''Perform segment on this small image'''
        #gradient = sobel(rgb2gray(self.cutedgeimage));
        segments = watershed(self.cutenforceedgeimage,markers=6, compactness=0.002,mask=self.themask);
        self.cutreseg=segments;
    
    def reSegment7(self):
        segments =quickshift(self.cutmaskedimage, kernel_size=3, max_dist=6, ratio=0.5);
        self.cutreseg=segments;
        
    def reSegment6(self):
        pass;
        inputimage=self.cutenforceedgeimage;
        inputmask=self.themask;
        inputmask=inputmask*10;
        
        #comparelist=[0.1, 0.05];
        
        comparelist=[0.1];
        for cpv in comparelist:
            pass;
            
            z1=inputimage>=cpv;
            z1=z1+0;
     
            
           
            label=21;
            
        
            taskimg=z1+inputmask;
            
            while (True):
                pass;
                    
                [hh,ll]=np.where(taskimg==10)
                
                if (len(hh)==0):
                    pass;
                    break;
                    
                v1=flood(taskimg,(hh[0],ll[0]));
                
                [hh2,ll2]=np.where(v1==True);
                
                taskimg[hh2,ll2]=label;
                label=label+1;
            
                
        
            
        
            
        [hss,lss]=np.shape(taskimg);
            
        while (True):
            [hh,ll]=np.where(taskimg==11)
            
            if (len(hh)==0):
                pass;
                break;
            clonetask=np.copy(taskimg);
            for (ii,jj) in zip(hh,ll):
                vv=self.getNeighborValue(taskimg,ii,jj,hss,lss);
                if (vv>0):
                    clonetask[ii,jj]=vv;
            taskimg=clonetask;
        taskimg=taskimg-20;
        taskimg=taskimg*(self.themask*1);
        
        
        
        
        cloneimg=np.copy(taskimg);
        
      
        '''
        for ii in range(hss):
            for jj in range(lss):
                if (taskimg[ii,jj]==0):
                    continue;
                vv=self.getNeighborLabel(taskimg,ii,jj,hss,lss);
                if (vv>0):
                    cloneimg[ii,jj]=vv;
        '''
               
        segments=cloneimg;
        
        self.cutreseg=segments;
        
        
        return label-20;
        
        
    def getNeighborValue(self, taskimg, ii,jj,hss,lss):
        pass;
        v1=-1;
        v2=-1;
        v3=-1;
        v4=-1;
        
        posh=ii-1;
        posl=jj;
        if (posh>0):
            v1=taskimg[posh,posl];
        
        posh=ii;
        posl=jj-1;
        if (posl>0):
            v2=taskimg[posh,posl];
      
        posh=ii+1;
        posl=jj;
        if (posh<hss):
            v3=taskimg[posh,posl];    
            
        posh=ii;
        posl=jj+1;
        if (posl<lss):
            v4=taskimg[posh,posl]; 
            
        if (v1>20):
            return v1;
        if (v2>20):
            return v2;
        if (v3>20):
            return v3;
        if (v4>20):
            return v4;
        return -1; 
    
    
    def getNeighborLabel(self, taskimg, ii, jj, hss, lss):
        pass;
        hstart=ii-1;
        hend=ii+2;
        lstart=jj-1;
        lend=jj+2;
        if (hstart<0):
            hstart=0;
        if (hend>hss):
            hend=hss;
        if (lstart<0):
            lstart=0;
        if (lend>lss):
            lend=lss;
        
        counter=0;
        maxv=np.max(taskimg);
        marray=np.zeros((maxv+1));
        for ii in range(hstart,hend):
            for jj in range(lstart,lend):
                value=taskimg[ii,jj];
                if (value<1):
                    continue;
                counter=counter+1;
                marray[value]=marray[value]+1;
        
        vv=np.max(marray);
        
        if ( (vv/(counter*1.0))<=0.5):
            return -1;
        
        return np.argmax(marray);
    
    
    
    def setMaskSegment(self):
        '''Filter Segment's based on self.cutid'''
        pass;
        [hh,ll]=np.where(self.cutseg==self.cutid);
        [nhh,nll]=np.where(self.cutseg!=self.cutid);
        self.cutresegmasked=np.copy(self.cutreseg);
        self.cutresegmasked[nhh,nll]=-1;
        value=self.cutresegmasked[hh,ll];
        self.maskedidlist=np.unique(value);
        
        
        
        
    
    def getEdgeValue(self, alpha=0.05):
        pass;
        [hss,lss]=np.shape(self.cutedgeimage);
        [hh,ll]=np.where(self.cutseg==self.cutid);
        htj=np.zeros((lss));
        htjcount=np.zeros((lss));
        ltj=np.zeros((hss));
        ltjcount=np.zeros((hss));
        for (hii,lii) in zip(hh,ll):
                           
            htj[lii]=htj[lii]+self.cutedgeimage[hii,lii];
            htjcount[lii]=htjcount[lii]+1;
            
            ltj[hii]=ltj[hii]+self.cutedgeimage[hii,lii];
            ltjcount[hii]=ltjcount[hii]+1;
        
        htj=htj/htjcount;
        ltj=ltj/ltjcount;
        stdhtj=np.std(htj);
        stdltj=np.std(ltj);
        htt=0;
        for ii in range(lss):
            #value=np.abs(htj[ii]-htj[ii+1]);
            value=htj[ii];
            if (value>=alpha):
                htt=htt+1;
        
        
        htt=htt*1.0/(lss);
        
        ltt=0;
        for ii in range(hss):
            #value=np.abs(ltj[ii]-ltj[ii+1]);
            value=ltj[ii];
            if (value>alpha):
                ltt=ltt+1;
                
        
        ltt=ltt*1.0/(hss);
        
        return [htt,ltt,stdhtj,stdltj];
    
    def getEdgeValue2(self):
        pass;
        [hh,ll]=np.where(self.cutseg==self.cutid);
        
        values=self.cutedgeimage[hh,ll];
        values2=self.cutenforceedgeimage[hh,ll];
        (vv,)=np.where(values2>0.4);
        pbordernum=len(vv);
        
        pedgemaxv=np.max(values);
        pcontentnum=len(hh);
        
        return [pedgemaxv,pcontentnum, pbordernum];
    
        
    
        
   
            
    def showMyImage(self):
        bmimg=mark_boundaries(self.cutimage,self.cutseg);
        imshow(bmimg);
        
    def showMyImage2(self):
        bmimg=mark_boundaries(self.cutedgeimage,self.cutseg);
        imshow(bmimg);   
        
    def showMyImage3(self):
        bmimg=mark_boundaries(self.cutenforceimage ,self.cutseg);
        imshow(bmimg); 
        
        
        
        
        
        
        
        
        

   
        
    
    
'''

      
filename="I:\\paper2020-2\\ccc.tif"
image=imread(filename);

di=DecsionImageHelper();
di.readImage(image);

imshow(di.image);

di2=di.clone();
di.image[1:100,:]=0;


        
        
'''     
'''dech2.imageSegTrans(segh);     '''
    
    










