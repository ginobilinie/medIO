    
'''
Target: Crop patches for kinds of medical images, such as hdr, nii, mha, mhd, raw and so on, and store them as hdf5 files
for multi-scale patches, the difficulty comes from the correspondence of different scale of patches
Created on Oct. 20, 2016
Author: Dong Nie 
'''



import SimpleITK as sitk

from multiprocessing import Pool
import os
import h5py
import numpy as np
from scipy import ndimage as nd

d1=32
d2=32
d3=32
dFA=[d1,d2,d3]
dSeg=[24,24,24]
step1=8
step2=10
step3=2
step=[step1,step2,step3]
    
'''
Actually, we donot need it any more,  this is useful to generate hdf5 database
'''
def cropCubic(matFA,matSeg,fileID,d,step,rate):
  
    eps=1e-5
    rate1=1.0/2
    rate2=1.0/4
    [row,col,leng]=matFA.shape
    cubicCnt=0
    estNum=10000
    trainFA=np.zeros([estNum,1, dFA[2],dFA[1],dFA[0]])
    trainFAScale1=np.zeros([estNum,1,int(rate1*dFA[2]),int(rate1*dFA[1]),int(rate1*dFA[0])])
    trainFAScale2=np.zeros([estNum,1,int(rate2*dFA[2]),int(rate2*dFA[1]),int(rate2*dFA[0])])
    trainSeg=np.zeros([estNum,1,dSeg[2],dSeg[1],dSeg[0]])
    trainSegScale1=np.zeros([estNum,1,int(rate1*dFA[2]),int(rate1*dFA[1]),int(rate1*dFA[0])])
    trainSegScale2=np.zeros([estNum,1,int(rate2*dFA[2]),int(rate2*dFA[1]),int(rate2*dFA[0])])
    print 'trainFA shape, ',trainFA.shape
    #to padding for input
    margin1=(dFA[0]-dSeg[0])/2
    margin2=(dFA[1]-dSeg[1])/2
    margin3=(dFA[2]-dSeg[2])/2
    cubicCnt=0
    marginD=[margin1,margin2,margin3]
    print 'matFA shape is ',matFA.shape
    matFAOut=np.zeros([row+2*marginD[0],col+2*marginD[1],leng+2*marginD[2]])
    print 'matFAOut shape is ',matFAOut.shape
    matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA
    matSegOut=np.zeros([row+2*marginD[0],col+2*marginD[1],leng+2*marginD[2]])
    matSegOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matSeg
    #for mageFA, enlarge it by padding
    if margin1!=0:
        matFAOut[0:marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA[marginD[0]-1::-1,:,:] #reverse 0:marginD[0]
        matFAOut[row+marginD[0]:matFAOut.shape[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA[matFA.shape[0]-1:row-marginD[0]-1:-1,:,:] #we'd better flip it along the 1st dimension
    if margin2!=0:
        matFAOut[marginD[0]:row+marginD[0],0:marginD[1],marginD[2]:leng+marginD[2]]=matFA[:,marginD[1]-1::-1,:] #we'd flip it along the 2nd dimension
        matFAOut[marginD[0]:row+marginD[0],col+marginD[1]:matFAOut.shape[1],marginD[2]:leng+marginD[2]]=matFA[:,matFA.shape[1]-1:col-marginD[1]-1:-1,:] #we'd flip it along the 2nd dimension
    if margin3!=0:
        matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],0:marginD[2]]=matFA[:,:,marginD[2]-1::-1] #we'd better flip it along the 3rd dimension
        matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]+leng:matFAOut.shape[2]]=matFA[:,:,matFA.shape[2]-1:leng-marginD[2]-1:-1]
    #for matseg, enlarge it by padding
    if margin1!=0:
        matSegOut[0:marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matSeg[marginD[0]-1::-1,:,:] #reverse 0:marginD[0]
        matSegOut[row+marginD[0]:matSegOut.shape[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matSeg[matSeg.shape[0]-1:row-marginD[0]-1:-1,:,:] #we'd better flip it along the 1st dimension
    if margin2!=0:
        matSegOut[marginD[0]:row+marginD[0],0:marginD[1],marginD[2]:leng+marginD[2]]=matSeg[:,marginD[1]-1::-1,:] #we'd flip it along the 2nd dimension
        matSegOut[marginD[0]:row+marginD[0],col+marginD[1]:matSegOut.shape[1],marginD[2]:leng+marginD[2]]=matSeg[:,matSeg.shape[1]-1:col-marginD[1]-1:-1,:] #we'd flip it along the 2nd dimension
    if margin3!=0:
        matSegOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],0:marginD[2]]=matSeg[:,:,marginD[2]-1::-1] #we'd better flip it along the 3rd dimension
        matSegOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]+leng:matSegOut.shape[2]]=matSeg[:,:,matSeg.shape[2]-1:leng-marginD[2]-1:-1]
        
    dsfactor = rate
    #matFAScale = nd.interpolation.zoom(matFA, zoom=dsfactor)
    matFAOutScale1 = nd.interpolation.zoom(matFAOut, zoom=rate1)
    #matFAOutScale1=np.zeros([mFAS1.shape[0]+1,mFAS1.shape[1]+1,mFAS1.shape[2]+1]) #by padding
    #matFAOutScale1[0:-1,0:-1,0:-1]=mFAS1
    matFAOutScale2 = nd.interpolation.zoom(matFAOut, zoom=rate2)
    #matFAOutScale2=np.zeros([mFAS2.shape[0]+1,mFAS2.shape[1]+1,mFAS2.shape[2]+1]) #by padding
    #matFAOutScale2[0:-1,0:-1,0:-1]=mFAS2
#     volOut1=sitk.GetImageFromArray(matFAOutScale1)
#     volOut2=sitk.GetImageFromArray(matFAOutScale2)
#     sitk.WriteImage(volOut2,'preMRSub%s_s8.nii'%fileID)
#     sitk.WriteImage(volOut1,'preMRSub%s_s16.nii'%fileID)
    matSegScale1 = nd.interpolation.zoom(matSegOut, zoom=rate1) 
    #matSegScale1=np.zeros([mSegS1.shape[0]+1,mSegS1.shape[1]+1,mSegS1.shape[2]+1]) #by padding
    #matSegScale1[0:-1,0:-1,0:-1]=mSegS1
    matSegScale2 = nd.interpolation.zoom(matSegOut, zoom=rate2)   
    #matSegScale2=np.zeros([mSegS2.shape[0]+1,mSegS2.shape[1]+1,mSegS2.shape[2]+1]) #by padding
    #matSegScale2[0:-1,0:-1,0:-1]=mSegS2
#     volOut1=sitk.GetImageFromArray(matSegScale1)
#     volOut2=sitk.GetImageFromArray(matSegScale2)
#     sitk.WriteImage(volOut2,'preSegSub%s_s8.nii'%fileID)
#     sitk.WriteImage(volOut1,'preSegSub%s_s16.nii'%fileID)
    print 'matSegScale1 shape, ',matSegScale1.shape 
    #fid=open('trainxxx_list.txt','a');
    resizeTimes1=1/rate1
    resizeTimes2=1/rate2
    for i in range(0,row-dSeg[0]+1,step[0]):
        for j in range(0,col-dSeg[1]+1,step[1]):
            for k in range(0,leng-dSeg[2]+1,step[2]):
                volSeg=matSeg[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]]
                if np.sum(volSeg)<eps:
                    continue
                cubicCnt=cubicCnt+1
                #index at scale 1
                iRate1=int(i/resizeTimes1)
                jRate1=int(j/resizeTimes1)
                kRate1=int(k/resizeTimes1)
                #index at scale 2
                iRate2=int(i/resizeTimes2)
                jRate2=int(j/resizeTimes2)
                kRate2=int(k/resizeTimes2)
                
                volFA=matFAOut[i:i+dFA[0],j:j+dFA[1],k:k+dFA[2]]
                volFAScale1=matFAOutScale1[iRate1:iRate1+int(rate1*dFA[0]),jRate1:jRate1+int(rate1*dFA[1]),
                                           kRate1:kRate1+int(rate1*dFA[2])] #note, 16*16*16
                volFAScale2=matFAOutScale2[iRate2:iRate2+int(rate2*dFA[0]),jRate2:jRate2+int(rate2*dFA[1]),
                                           kRate2:kRate2+int(rate2*dFA[2])] #note, 8*8*8
                volSegScale1=matSegScale1[iRate1:iRate1+int(rate1*dFA[0]),jRate1:jRate1+int(rate1*dFA[1]),
                                           kRate1:kRate1+int(rate1*dFA[2])] #note, here 16*16*16
                volSegScale2=matSegScale2[iRate2:iRate2+int(rate2*dFA[0]),jRate2:jRate2+int(rate2*dFA[1]),
                                           kRate2:kRate2+int(rate2*dFA[2])] #note, here 8*8*8
                trainFA[cubicCnt,0:,:,:]=volFA #32*32*32
                trainFAScale1[cubicCnt,0,:,:,:]=volFAScale1 #16*16*16
                trainFAScale2[cubicCnt,0,:,:,:]=volFAScale2 #8*8*8
                trainSeg[cubicCnt,0,:,:,:]=volSeg#24*24*24
                #print 'volSegScale1 shape, ',volSegScale1.shape
                trainSegScale1[cubicCnt,0,:,:,:]=volSegScale1 #16*16*16
                trainSegScale2[cubicCnt,0,:,:,:]=volSegScale2 #8*8*8

    trainFA=trainFA[0:cubicCnt,:,:,:,:]
    trainFAScale1=trainFAScale1[0:cubicCnt,:,:,:,:]
    trainFAScale2=trainFAScale2[0:cubicCnt,:,:,:,:]
    trainSeg=trainSeg[0:cubicCnt,:,:,:,:]
    trainSegScale1=trainSegScale1[0:cubicCnt,:,:,:,:]
    trainSegScale2=trainSegScale2[0:cubicCnt,:,:,:,:]
    with h5py.File('./trainMS_%s.h5'%fileID,'w') as f:
        f['dataMR32']=trainFA
        f['dataSeg24']=trainSeg
        f['dataMR16']=trainFAScale1
        f['dataMR8']=trainFAScale2
        f['dataSeg16']=trainSegScale1
        f['dataSeg8']=trainSegScale2
    with open('./trainMS_list.txt','a') as f:
        f.write('./trainMS_%s.h5\n'%fileID)
    return cubicCnt
    	
def main():
    path='/shenlab/lab_stor3/dongnie/prostate/'
    saveto='shenlab/lab_stor3/dongnie/prostate/'
   
    ids=[1,2,3,4,5,6,7,8,9,10,11]
    for id in ids:
        datafilename='prostate_%dto1_MRI.nii'%id #provide a sample name of your filename of data here
        datafn=os.path.join(path,datafilename)
        labelfilename='prostate_%dto1_CT.nii'%id  # provide a sample name of your filename of ground truth here
        labelfn=os.path.join(path,labelfilename)
        imgOrg=sitk.ReadImage(datafn)
        mrimg=sitk.GetArrayFromImage(imgOrg)
#       	mu=np.mean(mrimg)
#       	maxV=np.max(mrimg)
#       	minV=np.min(mrimg)
#       	mrimg=float(mrimg)
#       	mrimg=(mrimg-mu)/(maxV-minV)

        labelOrg=sitk.ReadImage(labelfn)
        labelimg=sitk.GetArrayFromImage(labelOrg) 
        #you can do what you want here for for your label img
        
        fileID='%d'%id
        rate=1
        cubicCnt=cropCubic(mrimg,labelimg,fileID,dFA,step,rate)
        print '# of patches is ', cubicCnt
    
if __name__ == '__main__':     
    main()
