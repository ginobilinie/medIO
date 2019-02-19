    
'''
Target: Crop patches for kinds of medical images, such as hdr, nii, mha, mhd, raw and so on, and store them as hdf5 files
for single-scale patches
Created on May 18, 2017
Author: Dong Nie 
'''



import SimpleITK as sitk

from multiprocessing import Pool
import os
import h5py
import numpy as np
from scipy import ndimage as nd

eps=1e-5
tol=0.95
d1=3
d2=168
d3=112
dFA=[d1,d2,d3] # size of patches of input data
dSeg=[1,168,112] # size of pathes of label data
step1=1
step2=8
step3=8
step=[step1,step2,step3]
    
'''
Actually, we donot need it any more,  this is useful to generate hdf5 database
'''
def extractMsPatches4OneSubject(matFA,matSeg,fileID,d,step,rate):
  
    rate1=1.0/2
    rate2=1.0/4
    [row,col,leng]=matFA.shape
    cubicCnt=0
    estNum=20000
    trainFA=np.zeros([estNum,1, dFA[0],dFA[1],dFA[2]],np.float16)
    trainSeg=np.zeros([estNum,1,dSeg[0],dSeg[1],dSeg[2]],dtype=np.int8)
    trainFA2D=np.zeros([estNum, dFA[0],dFA[1],dFA[2]],np.float16)
    trainFA2DScale1=np.zeros([estNum,int(rate1*dFA[0]),int(rate1*dFA[1]),int(rate1*dFA[2])])
    trainFA2DScale2=np.zeros([estNum,int(rate2*dFA[0]),int(rate2*dFA[1]),int(rate2*dFA[2])])
    trainSeg2D=np.zeros([estNum,dSeg[0],dSeg[1],dSeg[2]],dtype=np.int8)
    trainSeg2DScale1=np.zeros([estNum,int(rate1*dSeg[0]),int(rate1*dSeg[1]),int(rate1*dSeg[2])])
    trainSeg2DScale2=np.zeros([estNum,int(rate2*dSeg[0]),int(rate2*dSeg[1]),int(rate2*dSeg[2])])
    print 'trainFA shape, ',trainFA.shape
    #to padding for input, actually, it is not necessary to do the next step
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
    matSegScale1=np.rint(matSegScale1) 
    print np.unique(matSegOut)
    print np.unique(matSegScale1)
    #matSegScale1=np.zeros([mSegS1.shape[0]+1,mSegS1.shape[1]+1,mSegS1.shape[2]+1]) #by padding
    #matSegScale1[0:-1,0:-1,0:-1]=mSegS1
    matSegScale2 = nd.interpolation.zoom(matSegOut, zoom=rate2) 
    matSegScale2=np.rint(matSegScale2) 
     
    print np.unique(matSegScale2)
    #actually, we can specify a bounding box along the 2nd and 3rd dimension, so we can make it easier 
    resizeTimes1=1/rate1
    resizeTimes2=1/rate2
    for i in range(0,row-dSeg[0]+1,step[0]):
        for j in range(0,col-dSeg[1]+1,step[1]):
            for k in range(0,leng-dSeg[2]+1,step[2]):
                volSeg=matSeg[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]]
                if np.sum(volSeg)<eps:
                    continue
                #bgRate=1-np.count_nonzero(matSeg)/np.prod(matSeg.shape)
                #print bgRate
                #if bgRate>tol:
                #    continue;
                
                cubicCnt=cubicCnt+1
                #index at scale 1
            
                
#                 volFA=matFAOut[i:i+dFA[0],j:j+dFA[1],k:k+dFA[2]]
#               
#                 trainFA[cubicCnt,0,:,:,:]=volFA #32*32*32
#                 trainSeg[cubicCnt,0,:,:,:]=volSeg#24*24*24
# 
#                 trainFA2D[cubicCnt,:,:,:]=volFA #32*32*32
#                 trainSeg2D[cubicCnt,:,:,:]=volSeg#24*24*24
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
                trainFA[cubicCnt,0:,:,:]=volFA #for 3d
                trainFA2D[cubicCnt,:,:,:]=volFA #for 2d
                trainFA2DScale1[cubicCnt,:,:,:]=volFAScale1 #16*16*16
                trainFA2DScale2[cubicCnt,:,:,:]=volFAScale2 #8*8*8
                
                trainSeg[cubicCnt,0,:,:,:]=volSeg#for 3d
                trainSeg2D[cubicCnt,:,:,:]=volSeg#for 2d
                #print 'volSegScale1 shape, ',volSegScale1.shape
                trainSeg2DScale1[cubicCnt,:,:,:]=volSegScale1 #16*16*16
                trainSeg2DScale2[cubicCnt,:,:,:]=volSegScale2 #8*8*8

    #for 3d
    trainFA=trainFA[0:cubicCnt,:,:,:,:]
    trainSeg=trainSeg[0:cubicCnt,:,:,:,:]   
    
    #for 2d
    trainFA2D=trainFA2D[0:cubicCnt,:,:,:]
    trainFA2DScale1=trainFA2DScale1[0:cubicCnt,:,:,:]
    trainFA2DScale2=trainFA2DScale2[0:cubicCnt,:,:,:]
    
    #for 2d
    trainSeg2D=trainSeg2D[0:cubicCnt,:,:,:]
    trainSeg2DScale1=trainSeg2DScale1[0:cubicCnt,:,:,:]
    trainSeg2DScale2=trainSeg2DScale2[0:cubicCnt,:,:,:]

    with h5py.File('./train3x168x112ms_%s.h5'%fileID,'w') as f:
        f['dataMR']=trainFA
        f['dataSeg']=trainSeg
        f['dataMR2D']=trainFA2D
        f['dataSeg2D']=trainSeg2D
        f['dataMR2DScale1']=trainFA2DScale1
        f['dataSeg2DScale1']=trainSeg2DScale1
        f['dataMR2DScale2']=trainFA2DScale2
        f['dataSeg2DScale2']=trainSeg2DScale2
     
    with open('./trainPelvic3x168x112ms_all29_list.txt','a') as f:
        f.write('/home/dongnie/warehouse/mrs_data/train3x168x112ms_%s.h5\n'%fileID)
    return cubicCnt

#to remove zero slices along the 1st dimension
def stripNullSlices(tmpMR,mrimg,labelimg):
    startS=-1
    endS=-1
    for i in range(0,tmpMR.shape[0]):
        if np.sum(tmpMR[i,:,:])<eps:
            if startS==-1:
                continue
            else:
                endS=i-1
                break
        else:
            if startS==-1:
                startS=i
            else:
                continue
    if endS==-1: #means there is no null slices at the end
        endS=tmpMR.shape[0]-1
    return startS,endS
        
def main():
    path='/home/dongnie/warehouse/mrs_data/'
    saveto='/home/dongnie/warehouse/mrs_data/'
    #path='/shenlab/lab_stor3/dongnie/pelvicSeg/mrs_data/'
	#saveto='/shenlab/lab_stor3/dongnie/pelvicSeg/mrs_data/'
    ids=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    ids=[1,2,3,4,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
    for ind in ids:
        datafilename='img%d_crop.nii.gz'%ind #provide a sample name of your filename of data here
        datafn=os.path.join(path,datafilename)
        labelfilename='img%d_label_nie_crop.nii.gz'%ind  # provide a sample name of your filename of ground truth here
        labelfn=os.path.join(path,labelfilename)
        imgOrg=sitk.ReadImage(datafn)
        mrimg=sitk.GetArrayFromImage(imgOrg)
        tmpMR=mrimg
           #mrimg=mrimg

        labelOrg=sitk.ReadImage(labelfn)
        labelimg=sitk.GetArrayFromImage(labelOrg)
        if ind<14:
            labelimg=labelimg/10 
        print np.unique(labelimg)
           #mu=np.mean(labelimg)
           #maxV, minV=np.percentile(labelimg, [99 ,1])
          #labelimg=labelimg
           #labelimg=(labelimg-mu)/(maxV-minV)
        #you can do what you want here for for your label img
        
        rate=1
        print 'it comes to sub',ind
        print 'shape of mrimg, ',mrimg.shape
        startS,endS=stripNullSlices(tmpMR,mrimg,labelimg)
        print 'start slice is,',startS, 'end Slice is', endS
        mrimg=mrimg[startS:endS+1,:,:]
        print 'shape of mrimg, ',mrimg.shape
        originalMRI=mrimg
        dim2_start=35
        dim2_end=235
        dim3_start=80
        dim3_end=192
        mrimg=mrimg[:,dim2_start:dim2_end,dim3_start:dim3_end] #attention region
        volMRI=sitk.GetImageFromArray(mrimg)
        sitk.WriteImage(volMRI,'centerMRI_sub%d.nii.gz'%ind)
        mu=np.mean(mrimg)
        maxV, minV=np.percentile(mrimg, [99 ,1])
        print 'maxV,',maxV,' minV, ',minV
        mrimg=(mrimg-mu)/(maxV-minV)
        labelimg=labelimg[startS:endS+1,:,:]
        originalLabel=labelimg
        labelimg=labelimg[:,dim2_start:dim2_end,dim3_start:dim3_end]#attention region
        fileID='%d'%ind
        cubicCnt=extractMsPatches4OneSubject(mrimg,labelimg,fileID,dFA,step,rate)
        print '# of patches is ', cubicCnt

        tmpMat=mrimg
        tmpLabel=labelimg
        #reverse along the 1st dimension 
        mrimg=mrimg[tmpMat.shape[0]-1::-1,:,:]
        labelimg=labelimg[tmpLabel.shape[0]-1::-1,:,:]
        fileID='%d_flip1'%ind
        cubicCnt=extractMsPatches4OneSubject(mrimg,labelimg,fileID,dFA,step,rate)
        #reverse along the 2nd dimension 
        mrimg=mrimg[:,tmpMat.shape[1]-1::-1,:]
        labelimg=labelimg[:,tmpLabel.shape[1]-1::-1,:]
        fileID='%d_flip2'%ind
        cubicCnt=extractMsPatches4OneSubject(mrimg,labelimg,fileID,dFA,step,rate)
        #reverse along the 2nd dimension 
        mrimg=mrimg[:,:,tmpMat.shape[2]-1::-1]
        labelimg=labelimg[:,:,tmpLabel.shape[2]-1::-1]
        fileID='%d_flip3'%ind
        cubicCnt=extractMsPatches4OneSubject(mrimg,labelimg,fileID,dFA,step,rate)
if __name__ == '__main__':     
    main()
