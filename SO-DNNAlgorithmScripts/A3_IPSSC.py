# -*- coding: utf-8 -*-

'''
Run Algorithm 1: Initial patch-based sample set construction (IPSSC)
Generate Patch-based samples
'''
from CuterHelper import *;


[hss,lss,bds]=np.shape(image);

tmpresultimg=np.zeros((hss,lss,decisionnumber),dtype='float32');


segh=SegHelper();

segh.readfile(seghfile);

for ii in range(pointsarray.pointsnum):
    points=pointsarray.pointslist[ii];    
    print(points.pointnum,points.decsion)
    for hh,ll in points.data:
        print("%d----%d,%d"%(ii,hh,ll))
        theid=segh.imgseg[hh,ll];
        themember=segh.imgseglist[theid];
        themember['Decsion1']=points.decsion;
        allh= themember['Pixels-Row'];
        alll= themember['Pixels-Col'];
        
        pos=(int)(points.decsion);
        tmpresultimg[allh,alll,pos]=1;
        
    
'''获取结果图结束'''    

trainx=[];
trainy=[];



zerodecsion=np.zeros((decisionnumber),dtype='float32');
halfw=int(imagepatchwidth/2);

counter=0;

for ii in range(pointsarray.pointsnum):
    points=pointsarray.pointslist[ii];    
    print(points.pointnum,points.decsion)
    for hh,ll in points.data:
        suju=[0,hh-halfw,ll-halfw,imagepatchwidth];
        patch=CutFromImage(nmimage,suju);
        patch2=CutFromImage(tmpresultimg,suju);
        trainx.append(patch);  
        trainy.append(patch2);
    
resultx=np.stack(trainx,0);
resulty=np.stack(trainy,0);

[nns,hss,lss,bds]=np.shape(resultx);
selectrandom=np.random.permutation(nns); 

train_x=resultx[selectrandom,:];
train_y=resulty[selectrandom,:];


wf=open(IPSSCFile,'wb');
pickle.dump([train_x,train_y],wf);
wf.close();