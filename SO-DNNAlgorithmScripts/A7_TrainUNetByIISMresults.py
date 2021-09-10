# -*- coding: utf-8 -*-


'''
Train U-Net model by IISM's result samples;
'''


wf=open(IISMFile,'rb');
[train_x,train_y]=pickle.load(wf);
wf.close(); 

history=model.fit(train_x,train_y,batch_size=20,epochs=100);