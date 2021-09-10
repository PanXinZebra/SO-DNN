# -*- coding: utf-8 -*-

'''
Run FROM and IISM one iterate

run  of "A6" + "A7" will mean that the IISM algorithm is iterated once

'''



from CuterHelper import *;

segh=SegHelper();

segh.readfile(seghfile);


inputdata=np.zeros((1,imagepatchwidth,imagepatchwidth,bds),dtype='float32');
halfw=int(imagepatchwidth/2);

counter=0;

'''Category that want to eliminate from IISM algorithm, -1=eliminate none'''
exceptiocategorid=4;

for cc in segh.imgseglist.keys():
    member=segh.imgseglist[cc];
    member['token']=0;


for ii in range(pointsarray.pointsnum):
    points=pointsarray.pointslist[ii];    
    #print(points.pointnum,points.decsion)
    for hh,ll in points.data:
        theid=segh.imgseg[hh,ll];
        themember=segh.imgseglist[theid];
        themember['Decsion1']=points.decsion;
        themember['token']=1;
        


for cc in segh.imgseglist.keys():
    member=segh.imgseglist[cc];
    [hh,ll]=member["CenterPoint"];
    
    suju=[0,hh-halfw,ll-halfw,imagepatchwidth];
    patch1=CutFromImage(nmimage,suju);
    patch2=CutFromImageSingle(segh.imgseg,suju);
    
    inputdata[0,:,:,:]=patch1;
    
    rs=model.predict(inputdata);
    
    
    [hh,ll]=np.where(patch2==cc);
    
    size1=np.max(np.shape(hh))
    size2=np.max(np.shape(ll))
    if (size1==0 or size2==0):
        print("%d have on points"%cc)
        member["Fuzzyrepresentation"]=0;
        continue;
    
    resultdata=rs[0][hh,ll,:];
    
    fuzzydec=np.mean(resultdata,0);
    
    resultdata=np.argmax(resultdata,1);
    [idlist,numlist]=np.unique(resultdata[:],return_counts=True);
    print([idlist,numlist]);
    ps=np.argmax(numlist);
    pdr=idlist[ps];
    
    member["Fuzzyrepresentation"]=fuzzydec;


trainx=[];
trainy=[];

for ii in range(pointsarray.pointsnum):
    points=pointsarray.pointslist[ii];    
    print(points.pointnum,points.decsion)
    for hh,ll in points.data:
        pass;
        '''------------------Expand----------------------------------'''
        centerid=segh.imgseg[hh,ll];
        centermember=segh.imgseglist[centerid];
        centerdecsion=centermember['Decsion1'];
                
          
        suju=[0,hh-halfw,ll-halfw,imagepatchwidth];
        patch1=CutFromImage(nmimage,suju);
        patch2=CutFromImageSingle(segh.imgseg,suju);
        
        mylist=np.unique(patch2);
        
        havelist=[];
        havefuzzy=[];
        havedecsion=[];
        havadistance=[];
        
        for tmpid in mylist:
            smallmember=segh.imgseglist[tmpid];
            #print(smallmember['token']);
            if (smallmember['token']==1):
                havelist.append(tmpid);
                havefuzzy.append(smallmember['Fuzzyrepresentation']);
                havedecsion.append(smallmember['Decsion1']);
                havadistance.append(0);
        (havenum,)=np.shape(havadistance);
        decsionlist=[];
        for tmpid in mylist:
            if (tmpid==centerid):
                decsionlist.append(centermember['Decsion1']);
                continue;
            smallmember=segh.imgseglist[tmpid];
            if (smallmember['token']==1):
                decsionlist.append(smallmember['Decsion1']);
                continue;
            '''dis?'''
            if (centerdecsion==exceptiocategorid):
                decsionlist.append(-1);
                continue;
                
            
            for ii in range(havenum):
                pass;
                op1 = np.linalg.norm(havefuzzy[ii]-smallmember['Fuzzyrepresentation']);
                havadistance[ii]=op1;
            pos=np.argmin(havadistance);
            if (havadistance[pos]>aef):
                decsionlist.append(-1);
                continue;
            if (havedecsion[pos]==centerdecsion):
                decsionlist.append(centerdecsion);
            else:
                decsionlist.append(-1);
            
        
        resultpatch=np.zeros((imagepatchwidth,imagepatchwidth,decisionnumber),dtype='float32');
        
        for (ttid,ttdec) in zip(mylist,decsionlist):
            #print("%d %d"%(ttid,ttdec));
            (thh,tll)=np.where(patch2==ttid);
            if (ttdec==-1):
                continue;
            resultpatch[thh,tll,ttdec]=1;    
    
    
        '''-------------------------------------------------------'''
        trainx.append(patch1);  
        trainy.append(resultpatch);

resultx=np.stack(trainx,0);
resulty=np.stack(trainy,0);

[nns,hss,lss,bds]=np.shape(resultx);
selectrandom=np.random.permutation(nns); 

train_x=resultx[selectrandom,:];
train_y=resulty[selectrandom,:];


wf=open(IISMFile,'wb');
pickle.dump([train_x,train_y],wf);
wf.close();
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        