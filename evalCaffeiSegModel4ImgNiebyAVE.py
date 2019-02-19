    
'''
Target: evaluate your trained caffe model with the medical images. I use simpleITK to read medical images (hdr, nii, nii.gz, mha and so on)  
Created on Oct. 20, 2016
Author: Dong Nie 
Note, this is specified for the prostate, which input is larger than output
'''



import SimpleITK as sitk

from multiprocessing import Pool
import os
import h5py
import numpy as np  
import scipy.io as scio

# Make sure that caffe is on the python path:
#caffe_root = '/usr/local/caffe3/'  # this is the path in GPU server
caffe_root = '/home/dongnie/Desktop/Caffes/caffe/'  # this is the path in GPU server
import sys
sys.path.insert(0, caffe_root + 'python')
print caffe_root + 'python'
import caffe

caffe.set_device(0) #very important
caffe.set_mode_gpu()
### load the solver and create train and test nets
solver = None  # ignore this workaround for lmdb data (can't instantiate two solvers on the same data)
#solver = caffe.SGDSolver('infant_fcn_solver.prototxt') #for training
#protopath='/home/dongnie/caffe3D/examples/prostate/'
#protopath='/home/dongnie/Desktop/Caffes/caffe/examples/infantBrain32UNet/'
protopath='/home/dongnie/Desktop/Caffes/caffe/examples/iSeg/'
#mynet = caffe.Net(protopath+'infant_deploy_unet_bn_v2.prototxt',protopath+'infant_fcn_unet_bn_v2_iter_50000.caffemodel',caffe.TEST)
mynet = caffe.Net(protopath+'iSeg_deploy_23d_v1_random_res_dilation_better.prototxt',protopath+'iSeg_9sub_v1_random_res_dilation_better_0811_iter_100000.caffemodel',caffe.TEST)
print("blobs {}\nparams {}".format(mynet.blobs.keys(), mynet.params.keys()))

d1=32
d2=32
d3=32
dFA=[d1,d2,d3]
dSeg=[32,32,32]
step1=32
step2=32
step3=32
step=[step1,step2,step3]
    
def cropCubic(matFA,matT2,matSeg,fileID,d,step,rate):
    eps=1e-5
    #transpose
    matFA=np.transpose(matFA,(2,1,0))
    matT2=np.transpose(matT2,(2,1,0))
    matSeg=np.transpose(matSeg,(2,1,0))
    [row,col,leng]=matFA.shape
    margin1=(dFA[0]-dSeg[0])/2
    margin2=(dFA[1]-dSeg[1])/2
    margin3=(dFA[2]-dSeg[2])/2
    cubicCnt=0
    marginD=[margin1,margin2,margin3]
    #print 'matFA shape is ',matFA.shape
    matFAOut=np.zeros([row+2*marginD[0],col+2*marginD[1],leng+2*marginD[2]])
    #print 'matFAOut shape is ',matFAOut.shape
    matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA

#     matFAOut[0:marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA[0:marginD[0],:,:] #we'd better flip it along the first dimension
#     matFAOut[row+marginD[0]:matFAOut.shape[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA[row-marginD[0]:matFA.shape[0],:,:] #we'd better flip it along the 1st dimension
# 
#     matFAOut[marginD[0]:row+marginD[0],0:marginD[1],marginD[2]:leng+marginD[2]]=matFA[:,0:marginD[1],:] #we'd better flip it along the 2nd dimension
#     matFAOut[marginD[0]:row+marginD[0],col+marginD[1]:matFAOut.shape[1],marginD[2]:leng+marginD[2]]=matFA[:,col-marginD[1]:matFA.shape[1],:] #we'd better to flip it along the 2nd dimension
# 
#     matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],0:marginD[2]]=matFA[:,:,0:marginD[2]] #we'd better flip it along the 3rd dimension
#     matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]+leng:matFAOut.shape[2]]=matFA[:,:,leng-marginD[2]:matFA.shape[2]]
    if margin1!=0:
        matFAOut[0:marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA[marginD[0]-1::-1,:,:] #reverse 0:marginD[0]
        matFAOut[row+marginD[0]:matFAOut.shape[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA[matFA.shape[0]-1:row-marginD[0]-1:-1,:,:] #we'd better flip it along the 1st dimension
    if margin2!=0:
        matFAOut[marginD[0]:row+marginD[0],0:marginD[1],marginD[2]:leng+marginD[2]]=matFA[:,marginD[1]-1::-1,:] #we'd flip it along the 2nd dimension
        matFAOut[marginD[0]:row+marginD[0],col+marginD[1]:matFAOut.shape[1],marginD[2]:leng+marginD[2]]=matFA[:,matFA.shape[1]-1:col-marginD[1]-1:-1,:] #we'd flip it along the 2nd dimension
    if margin3!=0:
        matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],0:marginD[2]]=matFA[:,:,marginD[2]-1::-1] #we'd better flip it along the 3rd dimension
        matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]+leng:matFAOut.shape[2]]=matFA[:,:,matFA.shape[2]-1:leng-marginD[2]-1:-1]

    matOut=np.zeros((matSeg.shape[0],matSeg.shape[1],matSeg.shape[2]))
    used=np.zeros((matSeg.shape[0],matSeg.shape[1],matSeg.shape[2]))+eps
    cnt=0
    #fid=open('trainxxx_list.txt','a');
    for i in range(0,row-d[0]+1,step[0]):
        for j in range(0,col-d[1]+1,step[1]):
            for k in range(0,leng-d[2]+1,step[2]):
                volSeg=matSeg[i:i+d[0],j:j+d[1],k:k+d[2]]
                #print 'volSeg shape is ',volSeg.shape
                volFA=matFA[i:i+d[0]+2*marginD[0],j:j+d[1]+2*marginD[1],k:k+d[2]+2*marginD[2]]
                volT2=matT2[i:i+d[0]+2*marginD[0],j:j+d[1]+2*marginD[1],k:k+d[2]+2*marginD[2]]
                #print 'volFA shape is ',volFA.shape
                mynet.blobs['dataT1'].data[0,0,...]=volFA
                mynet.blobs['dataT2'].data[0,0,...]=volT2
                mynet.forward()
                temppremat = mynet.blobs['softmax'].data[0].argmax(axis=0) #Note you have add softmax layer in deploy prototxt
                #cnt=cnt+1
                #print 'tempremat shape is ',temppremat.shape
                #volOut=sitk.GetImageFromArray(temppremat)
                #sitk.WriteImage(volOut,'volOut1_%d.nii.gz'%cnt)
                #temppremat = mynet.blobs['conv3e'].data[0] #Note you have add softmax layer in deploy prototxt
                #temppremat=np.zeros([volSeg.shape[0],volSeg.shape[1],volSeg.shape[2]])
                matOut[i:i+d[0],j:j+d[1],k:k+d[2]]=matOut[i:i+d[0],j:j+d[1],k:k+d[2]]+temppremat;
                used[i:i+d[0],j:j+d[1],k:k+d[2]]=used[i:i+d[0],j:j+d[1],k:k+d[2]]+1;
    matOut=matOut/used
    matOut=np.rint(matOut)
    matOut=np.transpose(matOut,(2,1,0))
    matSeg=np.transpose(matSeg,(2,1,0))
    return matOut,matSeg

#this function is used to compute the dice ratio
def dice(im1, im2,tid):
    im1=im1==tid #make it boolean
    im2=im2==tid #make it boolean
    im1=np.asarray(im1).astype(np.bool)
    im2=np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    dsc=2. * intersection.sum() / (im1.sum() + im2.sum())
    return dsc

def main():
    #datapath='/home/dongnie/warehouse/xxx/'
    #datapath='/shenlab/lab_stor3/dongnie/prostate/'
    datapath='/home/dongnie/Desktop/Caffes/data/infantBrain/normals/'
    path='/home/dongnie/warehouse/iSeg/iSeg-2017-Training/'

    
    #ids=[1,2,3,4,5,6,7,8,9,10,11] 
    ids=[1,2,3,4,5,6,7,8,9]
    #files=os.listdir([datapath,'*.hdr']) 
    for i in range(0, len(ids)):
        ind=ids[i]    
        dataT1filename='subject-%d-T1.hdr'%ind #provide a sample name of your filename of data here
        dataT2filename='subject-%d-T2.hdr'%ind #provide a sample name of your filename of data here
        dataT1fn=os.path.join(path,dataT1filename)
        dataT2fn=os.path.join(path,dataT2filename)
        labelfilename='subject-%d-label.hdr'%ind  # provide a sample name of your filename of ground truth here
        labelfn=os.path.join(path,labelfilename)
        imgT1Org=sitk.ReadImage(dataT1fn)
        mrT1img=sitk.GetArrayFromImage(imgT1Org)
        tmpT1=mrT1img
        imgT2Org=sitk.ReadImage(dataT2fn)
        mrT2img=sitk.GetArrayFromImage(imgT2Org)
        #tmpMR=mrimg
        #mrimg=mrimg
        
        labelOrg=sitk.ReadImage(labelfn)
        labelimg=sitk.GetArrayFromImage(labelOrg)
        labelimg[labelimg>200]=3 #white matter
        labelimg[labelimg>100]=2 #gray matter
        labelimg[labelimg>4]=1 #csf
        #print 'labelimg elements are, ',np.unique(labelimg)
        #if ind<14:
        #   labelimg=labelimg/10
        #print np.unique(labelimg)
        #mu=np.mean(labelimg)
        #maxV, minV=np.percentile(labelimg, [99 ,1])
        #labelimg=labelimg
        #labelimg=(labelimg-mu)/(maxV-minV)
        #you can do what you want here for for your label img
        
        rate=1
        #print 'it comes to sub',ind
        #print 'shape of mrimg, ',mrT1img.shape
        
        mu1=np.mean(mrT1img)
        mu2=np.mean(mrT2img)
        #maxV, minV=np.percentile(mrimg, [99 ,1])
        #maxV1, minV1=np.ndarray.max(mrT1img)
        #print 'maxV1,',maxV1,' minV1, ',minV1
        std1=np.std(mrT1img)
        std2=np.std(mrT2img)
        mrT1img=(mrT1img-mu1)/std1
        mrT2img=(mrT2img-mu2)/std2
        #print 'maxV1, ',np.ndarray.max(mrT1img),' maxV2, ',np.ndarray.max(mrT2img)
        
        fileID='%d'%ind
        rate=1
        matOut,matSeg=cropCubic(mrT1img,mrT2img,labelimg,fileID,dSeg,step,rate)
        matOut[tmpT1==0]=0
        #print 'unique value of matOut is ',np.unique(matOut)
        dr1=dice(matOut,matSeg,1)
        dr2=dice(matOut,matSeg,2)
        dr3=dice(matOut,matSeg,3)
        #print 'dr1: ',dr1,'dr2: ',dr2,'dr3: ',dr3
        volOut=sitk.GetImageFromArray(matOut)
        sitk.WriteImage(volOut,'preSub%d.nii.gz'%ind)
        volSeg=sitk.GetImageFromArray(matSeg)
        sitk.WriteImage(volSeg,'gt%d.nii.gz'%ind)
        #np.save('preSub'+fileID+'.npy',matOut)
        # here you can make it round to nearest integer 
        #now we can compute dice ratio
    
if __name__ == '__main__':     
    main()
