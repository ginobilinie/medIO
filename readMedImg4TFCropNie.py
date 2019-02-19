    
'''
Target: Crop patches for kinds of medical images, such as hdr, nii, mha, mhd, raw and so on, and store them as hdf5 files
Created on Oct. 20, 2016
Author: Dong Nie 
'''



import SimpleITK as sitk

from multiprocessing import Pool
import os
import h5py
import numpy as np  
import tensorflow as tf

d1=32
d2=32
d3=32
dFA=[d1,d2,d3]
dSeg=[24,24,24]
step1=10
step2=10
step3=8
step=[step1,step2,step3]

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
'''
Actually, we donot need it any more,  this is useful to generate hdf5 database
'''
def cropCubic(matFA,matSeg,fileID,d,step,rate):
    eps=1e-5
    [row,col,len]=matFA.shape
    cubicCnt=0
    trainFA=[]
    trainSeg=[]
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
    if margin1!=0:
        matFAOut[0:marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA[marginD[0]-1::-1,:,:] #reverse 0:marginD[0]
        matFAOut[row+marginD[0]:matFAOut.shape[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA[matFA.shape[0]-1:row-marginD[0]-1:-1,:,:] #we'd better flip it along the 1st dimension
    if margin2!=0:
        matFAOut[marginD[0]:row+marginD[0],0:marginD[1],marginD[2]:leng+marginD[2]]=matFA[:,marginD[1]-1::-1,:] #we'd flip it along the 2nd dimension
        matFAOut[marginD[0]:row+marginD[0],col+marginD[1]:matFAOut.shape[1],marginD[2]:leng+marginD[2]]=matFA[:,matFA.shape[1]-1:col-marginD[1]-1:-1,:] #we'd flip it along the 2nd dimension
    if margin3!=0:
        matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],0:marginD[2]]=matFA[:,:,marginD[2]-1::-1] #we'd better flip it along the 3rd dimension
        matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]+leng:matFAOut.shape[2]]=matFA[:,:,matFA.shape[2]-1:leng-marginD[2]-1:-1]
        
       
    #fid=open('trainxxx_list.txt','a');
    for i in range(0,row-dSeg[0],step[0]):
        for j in range(0,col-dSeg[1],step[1]):
            for k in range(0,len-dSeg[2],step[2]):
                volSeg=float(matSeg[i:i+dSeg[0],j:j+dSeg[1],k:k+d[2]])
                if np.sum(volSeg)<eps:
                    continue
                cubicCnt=cubicCnt+1
                volFA=float(matFA[i:i+dFA[0],j:j+dFA[1],k:k+dFA[2]])
                trainFA[:,:,:,1,cubicCnt]=volFA
                trainSeg[:,:,:,1,cubicCnt]=volSeg
    trainFA=float(trainFA)
    trainSeg=float(trainSeg)
    with h5py.File('./train_%s.h5'%fileID,'w') as f:
        f['dataMR']=trainFA
        f['dataSeg']=trainSeg
    with open('./trainxxx_list.txt','a') as f:
        f.write('./trainxxx.h5\n')
    return cubicCnt
 
'''
    Here, we write code to crop patches for more than 1 subjects and generate TFrecords formate dataset
'''
def cropCubic4MoreSub(dataPath,segPath,fileIDs,step,savePath,saveFName,rate):
    eps=1e-5
    cubicCnt=0
    
    output_fname = os.path.join(savePath, saveFName + '.tfrecords')
    print("output_fname: %s" % output_fname);
    
    writer = tf.python_io.TFRecordWriter(output_fname)

    for ind in fileIDs:
        datafilename='prostate_%dto1_MRI.nii'%ind #provide a sample name of your filename of data here
        datafn=os.path.join(dataPath,datafilename)
        labelfilename='prostate_%dto1_CT.nii'%ind  # provide a sample name of your filename of ground truth here
        labelfn=os.path.join(segPath,labelfilename)
        imgOrg=sitk.ReadImage(datafn)
        matFA=sitk.GetArrayFromImage(imgOrg)
        labelOrg=sitk.ReadImage(labelfn)
        matSeg=sitk.GetArrayFromImage(labelOrg)
        #if we need some preprocessing, we can add codes here
        [row,col,leng]=matFA.shape
        margin1=(dFA[0]-dSeg[0])/2
        margin2=(dFA[1]-dSeg[1])/2
        margin3=(dFA[2]-dSeg[2])/2
        cubicCnt=0
        marginD=[margin1,margin2,margin3]
        print 'matFA shape is ',matFA.shape
        matFAOut=np.zeros([row+2*marginD[0],col+2*marginD[1],leng+2*marginD[2]])
        print 'matFAOut shape is ',matFAOut.shape
        matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA
        if margin1!=0:
            matFAOut[0:marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA[marginD[0]-1::-1,:,:] #reverse 0:marginD[0]
            matFAOut[row+marginD[0]:matFAOut.shape[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA[matFA.shape[0]-1:row-marginD[0]-1:-1,:,:] #we'd better flip it along the 1st dimension
        if margin2!=0:
            matFAOut[marginD[0]:row+marginD[0],0:marginD[1],marginD[2]:leng+marginD[2]]=matFA[:,marginD[1]-1::-1,:] #we'd flip it along the 2nd dimension
            matFAOut[marginD[0]:row+marginD[0],col+marginD[1]:matFAOut.shape[1],marginD[2]:leng+marginD[2]]=matFA[:,matFA.shape[1]-1:col-marginD[1]-1:-1,:] #we'd flip it along the 2nd dimension
        if margin3!=0:
            matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],0:marginD[2]]=matFA[:,:,marginD[2]-1::-1] #we'd better flip it along the 3rd dimension
            matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]+leng:matFAOut.shape[2]]=matFA[:,:,matFA.shape[2]-1:leng-marginD[2]-1:-1]
        #for test
#         myOut=sitk.GetImageFromArray(matFAOut)
#         sitk.WriteImage(myOut,'enlargeSub%d.nii.gz'%ind)
        
        for i in range(0,row-dSeg[0],step[0]):
            for j in range(0,col-dSeg[1],step[1]):
                for k in range(0,leng-dSeg[2],step[2]):
                    volSeg=matSeg[i:i+dSeg[0]-1,j:j+dSeg[1]-1,k:k+dSeg[2]-1]
                    if np.sum(volSeg)<eps:
                        continue
                    volFA=matFAOut[i:i+dFA[0]-1,j:j+dFA[1]-1,k:k+dFA[2]-1]
                    cubicCnt=cubicCnt+1
                    image_raw=volFA.tostring()
                    label_raw=volSeg.tostring()
#                     image_raw=volFA
#                     image_raw=image_raw.astype(np.float32)
#                     label_raw=volSeg
#                     label_raw=label_raw.astype(np.float32)
                    sizeFA = volFA.shape
                    sizeSeg = volSeg.shape
       
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'dataHeight': _int64_feature(sizeFA[0]),
                        'dataWidth': _int64_feature(sizeFA[1]),
                        'dataDepth': _int64_feature(sizeFA[2]),
                        'labelHeight': _int64_feature(sizeFA[0]),
                        'labelWidth': _int64_feature(sizeFA[1]),
                        'labelDepth': _int64_feature(sizeFA[2]),
                        'label_raw': _bytes_feature(label_raw),
                        'image_raw': _bytes_feature(image_raw)}))
                    writer.write(example.SerializeToString())
    return cubicCnt
        
def main():
    path='/home/dongnie/warehouse/prostate/'
    saveto='/home/dongnie/warehouse/prostate'
    caffeApp=0
    fileIDs=[1,2,3,4,5,6,7,8,9,10,11]
    tfApp=1-caffeApp
    if caffeApp:
        datafilename='prostate_%dto1_MRI.nii'%id #provide a sample name of your filename of data here
        datafn=os.path.join(path,datafilename)
        labelfilename='prostate_%dto1_CT.nii'%id  # provide a sample name of your filename of ground truth here
        labelfn=os.path.join(path,labelfilename)
        imgOrg=sitk.ReadImage(datafn)
        mrimg=sitk.GetArrayFromImage(imgOrg)
    
    #       mu=np.mean(mrimg)
    #       maxV=np.max(mrimg)
    #       minV=np.min(mrimg)
    #       mrimg=float(mrimg)
    #       mrimg=(mrimg-mu)/(maxV-minV)
        #img=sitk.GetImageFromArray(mrimg)
        #img.SetSpacing(imgtmp.GetSpacing())
        #img.SetOrigin(imgtmp.GetOrigin())
        #img.SetDirection(imgtmp.GetDirection())
        #normfilter=sitk.NormalizeImageFilter()
        #img=normfilter.Execute(img)
        labelOrg=sitk.ReadImage(labelfn)
        labelimg=sitk.GetArrayFromImage(labelOrg) 
        #you can do what you want here for for your label img
        
        fileID='%d'%id
        rate=1
        cubicCnt=cropCubic(mrimg,labelimg,fileID,step,rate)
    elif tfApp:
        saveFName='trainAllSub'
        rate=1
        cropCubic4MoreSub(path, path, fileIDs,step,saveto,saveFName,rate)
    
if __name__ == '__main__':     
    main()
