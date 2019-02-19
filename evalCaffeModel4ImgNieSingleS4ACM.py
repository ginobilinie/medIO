    
'''
Target: evaluate your trained caffe model with the medical images. I use simpleITK to read medical images (hdr, nii, nii.gz, mha and so on)  
Created on Oct. 20, 2016
Author: Dong Nie 
Note, this is specified for the prostate, which input is larger than output
Also, this can be used to generate mutli-scale input/output, e.g., 8/16/32 and so on
'''



import SimpleITK as sitk

from multiprocessing import Pool
import os
import h5py
import numpy as np  
import scipy.io as scio
from scipy import ndimage as nd
from imgUtils import psnr
from imgUtils import dice

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
protopath='/home/dongnie/Desktop/Caffes/caffe/examples/pelvicSeg/'
#mynet = caffe.Net(protopath+'pelvic_deploy_3d_v2.prototxt',protopath+'pelvic_fcn_v2_iter_100000.caffemodel',caffe.TEST)
#mynet = caffe.Net(protopath+'pelvic_deploy_3d_v2.prototxt',protopath+'pelvic_3d_v2_iter_100000.caffemodel',caffe.TEST)
#mynet = caffe.Net(protopath+'pelvic_deploy_23d_v5_res.prototxt',protopath+'pelvic32_v5_res_iter_100000.caffemodel',caffe.TEST)
mynet = caffe.Net(protopath+'pelvic_deploy_23d_v5_res.prototxt',protopath+'pelvic32_v5_res_iter_100000.caffemodel',caffe.TEST)
print("blobs {}\nparams {}".format(mynet.blobs.keys(), mynet.params.keys()))

types=4
d1=3 #16 #8 #32
d2=168 #16 #8 #32
d3=112 #16  #8 #32
dFA=[d1,d2,d3]
#dSeg=[24,24,24]
#dSeg=[8,8,8]
dSeg=[1,168,112]
#dSeg=[112,168,5]
step1=1
step2=8
step3=8
step=[step1,step2,step3]
    
def cropCubic(matFA,matSeg,fileID,d,step,rate):
    eps=1e-5
	#transpose
    #matFA=np.transpose(matFA,(2,1,0))
    #matSeg=np.transpose(matSeg,(2,1,0))
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
    #matFAOutScale = nd.interpolation.zoom(matFAOut, zoom=rate)
    #matSegScale=nd.interpolation.zoom(matSeg, zoom=rate)
    dim1=np.arange(80,192)
    dim2=np.arange(35,235)
    x1=80
    x2=192
    y1=35
    y2=235
    #matFAOutScale = matFAOut[:,y1:y2,x1:x2] #note, matFA and matFAOut same size 
    #matSegScale = matSeg[:,y1:y2,x1:x2]
    matFAOutScale = matFAOut[:,y1:y2,x1:x2] #note, matFA and matFAOut same size 
    matSegScale = matSeg[:,y1:y2,x1:x2]
    #matFAOutScale = matFAOut #note, matFA and matFAOut same size 
    #matSegScale = matSeg

    matOut=np.zeros((matSegScale.shape[0],matSegScale.shape[1],matSegScale.shape[2]),dtype=np.float)
    used=np.zeros((matSegScale.shape[0],matSegScale.shape[1],matSegScale.shape[2]),dtype=np.float)+eps 
    matOuts=np.zeros((types, matSegScale.shape[0],matSegScale.shape[1],matSegScale.shape[2]),dtype=np.float) #for prob maps
    useds=np.zeros((types, matSegScale.shape[0],matSegScale.shape[1],matSegScale.shape[2]),dtype=np.float)+eps  #for prob maps
    [row,col,leng]=matSegScale.shape
        
    #fid=open('trainxxx_list.txt','a');
    for i in range(0,row-d[0]+1,step[0]):
        for j in range(0,col-d[1]+1,step[1]):
            for k in range(0,leng-d[2]+1,step[2]):
                volSeg=matSeg[i:i+d[0],j:j+d[1],k:k+d[2]]
                #print 'volSeg shape is ',volSeg.shape
                volFA=matFAOutScale[i:i+d[0]+2*marginD[0],j:j+d[1]+2*marginD[1],k:k+d[2]+2*marginD[2]]
                #print 'volFA shape is ',volFA.shape,'k and k+ is',k,k+d[2]+2*marginD[2]
                mynet.blobs['dataMR2D'].data[0,...]=volFA
                mynet.forward()
                #temppremat = mynet.blobs['softmax'].data[0].argmax(axis=0) #Note you have add softmax layer in deploy prototxt
                temppremat = mynet.blobs['softmax'].data[0] #Note you have add softmax layer in deploy prototxt
                probMaps=temppremat #for prob maps
                temppremat = np.argmax(temppremat,axis=0)  #Note you have add softmax layer in deploy prototxt
                #temppremat=np.zeros([volSeg.shape[0],volSeg.shape[1],volSeg.shape[2]])
                matOut[i:i+d[0],j:j+d[1],k:k+d[2]]=matOut[i:i+d[0],j:j+d[1],k:k+d[2]]+temppremat
                used[i:i+d[0],j:j+d[1],k:k+d[2]]=used[i:i+d[0],j:j+d[1],k:k+d[2]]+1
                matOuts[i:i+d[0],j:j+d[1],k:k+d[2]]=matOuts[i:i+d[0],j:j+d[1],k:k+d[2]]+probMaps #for prob maps
                useds[i:i+d[0],j:j+d[1],k:k+d[2]]=useds[i:i+d[0],j:j+d[1],k:k+d[2]]+1 # for prob maps
    matOut=matOut/used
    matOut=np.rint(matOut)
    matOuts=matOuts/useds
    #print 'matOut element, ',np.unique(matOut)
    #matOut=np.transpose(matOut,(2,1,0))
    #matSegScale=np.transpose(matSegScale,(2,1,0))
    #print 'matSeg.shape, ',matSeg.shape[0],matSeg.shape[1],matSeg.shape[2]
    matOut1=np.zeros([matSeg.shape[0],matSeg.shape[1],matSeg.shape[2]],dtype=np.float)
    matOut1[:,y1:y2,x1:x2]=matOut
    matProb=np.zeros([types, matSeg.shape[0],matSeg.shape[1],matSeg.shape[2]],dtype=np.float) #for prob maps
    matProb[:,:,y1:y2,x1:x2]=matProb #for prob maps
    #matOut1=np.transpose(matOut1,(2,1,0))
    #matSeg=np.transpose(matSeg,(2,1,0))
    return matOut1,matSeg,matProb

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
    dsc=0
    if (im1.sum()+im2.sum())>0:
        dsc=2. * intersection.sum() / (im1.sum() + im2.sum())
    return dsc

def main():
    datapath='/home/dongnie/warehouse/mrs_data/'
    #datapath='/shenlab/lab_stor3/dongnie/prostate/'
    #datapath='/shenlab/lab_stor3/dongnie/mrs/' 
    ids=[1,2,3,4,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29] 
    ids=[1,2,3,4,6,7,8,10,11,12,13] 
    #ids=[2,3,4,5,6,7,8]
    dc0=np.zeros(len(ids)) 
    dc1=np.zeros(len(ids)) 
    dc2=np.zeros(len(ids)) 
    dc3=np.zeros(len(ids)) 
    for i in range(0, len(ids)):
        myid=ids[i]    
        datafilename='img%d_crop.nii.gz'%myid
        datafn=os.path.join(datapath,datafilename)
        labelfilename='img%d_label_nie_crop.nii.gz'%myid  # provide a sample name of your filename of ground truth here
        labelfn=os.path.join(datapath,labelfilename)
        imgOrg=sitk.ReadImage(datafn)
        mrimg=sitk.GetArrayFromImage(imgOrg)
        mu=np.mean(mrimg)
        maxV=np.amax(mrimg)
        minV=np.amin(mrimg)
# 		print mrimg.dtype
#  		#mrimg=float(mrimg)
        mrimg=(mrimg-mu)/(maxV-minV)
        labelOrg=sitk.ReadImage(labelfn)
        labelimg=sitk.GetArrayFromImage(labelOrg) 
        #you can do what you want here for for your label img
        
        fileID='%d'%myid
        rate=1.0/1
        matOut,matSeg,matProb=cropCubic(mrimg,labelimg,fileID,dSeg,step,rate)
        matOut=np.rint(matOut)
        if i<14:
            matSeg=np.rint(matSeg/10)
        #print 'matOUt element: ',np.unique(matOut)
        #print 'matSeg element: ',np.unique(matSeg)
        pr0=dice(matOut,matSeg,0)
        #print 'dice is, ',pr0
        pr1=dice(matOut,matSeg,1)
        #print 'dice is, ',pr1
        pr2=dice(matOut,matSeg,2)
        #print 'dice is, ',pr2
        pr3=dice(matOut,matSeg,3)
        print 'dice for sub ',myid,' is ', pr0,' ',pr1,' ',pr2,' ',pr3
        volOut=sitk.GetImageFromArray(matOut)
        sitk.WriteImage(volOut,'preSub%d_3x168x112_res_v5_11sub.nii'%myid)
        volSeg=sitk.GetImageFromArray(matSeg)
        sitk.WriteImage(volSeg,'gt%d.nii'%myid)
        volProb=sitk.GetImageFromArray(matProb[0,:,:,:])
        sitk.WriteImage(volProb,'preSub%d_prob0_3x168x112_res_v5_11sub.nii'%myid)
        volProb=sitk.GetImageFromArray(matProb[1,:,:,:])
        sitk.WriteImage(volProb,'preSub%d_prob1_3x168x112_res_v5_11sub.nii'%myid)
        volProb=sitk.GetImageFromArray(matProb[2,:,:,:])
        sitk.WriteImage(volProb,'preSub%d_prob2_3x168x112_res_v5_11sub.nii'%myid)
        volProb=sitk.GetImageFromArray(matProb[3,:,:,:])
        sitk.WriteImage(volProb,'preSub%d_prob3_3x168x112_res_v5_11sub.nii'%myid)
        #np.save('preSub'+fileID+'.npy',matOut)
        # here you can make it round to nearest integer 
        #now we can compute dice ratio

if __name__ == '__main__':     
    main()
