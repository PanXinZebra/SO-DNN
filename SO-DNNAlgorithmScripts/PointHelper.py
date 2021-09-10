# -*- coding: utf-8 -*-

import numpy as np;
from scipy.io import loadmat;
import sys
import re;
import string;

class DFPoints:
    filename='';

    decsion=0;
    
    data=None; 
    
    pointnum=0;


    
    def __init__(self,fn,dc):
        self.filename=fn;
        self.decsion=dc;

    
    def read(self):
        if (self.filename[-4:].lower()=='.mat'):
            self.readMAT();
            
        
        if (self.filename[-4:].lower()=='.ptp'):
            self.readPTP();
        else:
            self.readRoiTextFile();
            
            
    
    def readMAT(self):
        pass;
        self.data=(loadmat(self.filename))['sample'];
        (self.pointnum,dim)=np.shape(self.data);
        
        x1=self.data[:,1];
        x2=self.data[:,0];
        self.data=np.stack( (x1,x2),1);
    
    
    
    def readPTP(self):
        pass;
        file_object = open(self.filename);
        try:
             all_the_text = file_object.read( )
        finally:
             file_object.close()
             
        lines=all_the_text.splitlines();
        #linenum=string.atoi(lines[0]);
        linenum=int(lines[0]);
        mylist=np.zeros([linenum,2],dtype='int32');
        
        for i in range(1,linenum+1):
            tline=re.split(' ',lines[i]);
            row=-1;
            col=-1;
            #row=string.atoi(tline[0]);
            #col=string.atoi(tline[1]);
            row=int(tline[0]);
            col=int(tline[1]);
            mylist[i-1]=[row,col];
        self.data=mylist;
        (self.pointnum,dim)=np.shape(self.data);
            
    
    def readRoiTextFile(self):
        file_object = open(self.filename);
        try:
             all_the_text = file_object.read( )
        finally:
             file_object.close()
             
        lines=all_the_text.splitlines();
        num=0;
        mylist=np.zeros([len(lines)-8,2],dtype='int32');
        for i in range(0,len(lines)):
            tline=re.split(' |\t',lines[i]);
            x=-1;
            y=-1;
            t=0;
            if len(tline[0])!=0:
                continue;
            for kk in tline:
                if (len(kk)==0):
                    continue;
                t=t+1;
                if t==2:
                    x=string.atoi(kk);        
                if t==3:
                    y=string.atoi(kk);     
            if x!=-1 and y!=-1:
                hs=y;
                ls=x;
                num=num+1;
                mylist[num-1]=[hs,ls];
        self.data=mylist;
        (self.pointnum,dim)=np.shape(self.data);
    
    
    def randomNumlist(self, maxnum, num):
        x=np.random.random(maxnum)
        ww=np.argsort(x)
        return ww[0:num];
    
    
    
    def getPointList(self,num,random=1):
        mlist=[];
        if (random==1):    
            if (num<=(self.pointnum)):
                idx=self.randomNumlist(self.pointnum,num);
            else:
                idx=self.randomNumlist(self.pointnum,self.pointnum);
                nn=num-self.pointnum;
                while (nn>0):
                    idx2=self.randomNumlist(self.pointnum,nn);
                    idx=np.concatenate([idx,idx2],0);
                    nn=nn-len(idx2);
        else:
            if (num<=(self.pointnum)):
                idx=range(num);
            else:
                idx=range(self.pointnum);
                nn=num-self.pointnum;
                while (nn>0):
                    if (nn>=self.pointnum):
                        idx2=range(self.pointnum);
                    else:
                        idx2=range(nn);
                    idx.extend(idx2);
                    nn=nn-len(idx2);
        
        if (num==-1):
            idx=range(self.pointnum);
        
        for ii in idx:
            pt=(self.data[ii,0],self.data[ii,1]);
            mlist.append(pt);
        return mlist;
                
            
    def toString(self):
        print(" Size="+str(self.pointnum)+" Decision="+str(self.decsion));


    
    def enlarge(self,eng):
        pass;
        self.data=self.data*eng;
        

class DFPointsArray:
    
    pointslist=[];
    decisionlist=[];
    pointsnum=0; '''pointfile文件的个数'''
        
    
    def __init__(self, samplelist=None):
        self.pointslist=[]; 
        self.decisionlist=[];
        if samplelist==None:
            pass;
            return;
            
        n=int(len(samplelist)/2);
        for i in range(0,n):
            points=DFPoints(samplelist[i*2],samplelist[i*2+1]);
            points.read();
            self.pointslist.append(points);
            self.decisionlist.append(points.decsion);
        self.pointsnum=n;
    
    def append(self,points):
        self.pointslist.append(points);
        self.decisionlist.append(points.decsion);
        self.pointsnum=self.pointsnum+1;
        
    
    def getSubArray(self,num,random=1):
        sarray=DFPointsArray();
        for i in range(0,self.pointsnum):
            pt=self.pointslist[i].getSubPoints(num,random);
            sarray.append(pt);
        return sarray;


    def getDecsionNum(self):
        maxv=max(self.decisionlist);
        return maxv+1;
    
    
    
    def createBinaryDecision(self,num=-1):
        maxv=max(self.decisionlist);
        mylist=[];
        for i in range(0,self.pointsnum):
            ptnum=self.pointslist[i].pointnum;
            if (num>=0):
                ptnum=num;
            zz=np.zeros((ptnum,maxv+1),dtype='uint8');
            zz[:, self.pointslist[i].decsion]=zz[:, self.pointslist[i].decsion]+1;
            mylist.append(zz);
            
        return np.concatenate(mylist,0)
    
    
    def createBinaryDecisionDouble(self,num=-1):
        maxv=max(self.decisionlist);
        mylist=[];
        for i in range(0,self.pointsnum):
            ptnum=self.pointslist[i].pointnum;
            if (num>=0):
                ptnum=num;
            zz=np.zeros((ptnum,maxv+1),dtype='uint8');
            zz[:, self.pointslist[i].decsion]=zz[:, self.pointslist[i].decsion]+1;
            mylist.append(zz);
            mylist.append(zz);
            
        return np.concatenate(mylist,0)
    
    def toString(self):
        for i in range(0,self.pointsnum):
            self.pointslist[i].toString();

    
    def enlarge(self,eng):
        pass;
        for kk in self.pointslist:
            kk.enlarge(eng);
        
        