# -*- coding: utf-8 -*-

'''
Perform 
Algorithm 4: Segment classification algorithm (SCA)
obtain classfy image;
'''


inputdata=np.zeros((1,hss,lss,bds),dtype='float32');
halfw=int(imagepatchwidth/2);

counter=0;

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
        continue;
    
    
    resultdata=rs[0][hh,ll,:];
    
    fuzzydec=np.mean(resultdata,0);
    
    resultdata=np.argmax(resultdata,1);
    [idlist,numlist]=np.unique(resultdata[:],return_counts=True);
    print([idlist,numlist]);
    ps=np.argmax(numlist);
    pdr=idlist[ps];
    print(cc);
    
   
    member["Decsion0"]=pdr;
    print(cc);


segh.saveColoredDecsion(resultimage,colors);