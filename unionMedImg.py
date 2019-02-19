    
'''
Target: avg for kinds of medical images, such as hdr, nii, mha, mhd, raw and so on, and store them as hdf5 files
Created on Oct. 20, 2016
Author: Dong Nie 
'''



import SimpleITK as sitk

from multiprocessing import Pool
import os
import h5py
import numpy as np  
from imgUtils import dice

def mostFreqElement(arr):
    arr0=(arr==0)
    arr1=(arr==1)
    arr2=(arr==2)
    arr3=(arr==3)
    l0=len(arr0)
    l1=len(arr1)
    l2=len(arr2)
    l3=len(arr3)
    res=2
    if l3>l2 and l3>l1 and l3>l0:
        res=3
    if l2>l3 and l2>l1 and l2>l0:
        res=2
    if l1>l3 and l1>l2 and l1>l0:
        res=1
    if l0>l3 and l0>l2 and l0>l1:
        res=0
    return res
def main():
    path='./'
    saveto='./'
    ids=[1,2,3,4,6,7,8,10,11,12,13]
    for ind in ids:
        datafilename='preSub%d_denoised_v1.nii'%ind #provide a sample name of your filename of data here
        datafn=os.path.join(path,datafilename)
        imgOrg=sitk.ReadImage(datafn)
        img1=sitk.GetArrayFromImage(imgOrg)
        
        
        labelfilename='preSub%d_denoised_v1_5x168x112.nii'%ind  # provide a sample name of your filename of ground truth here
        labelfn=os.path.join(path,labelfilename)
        imgOrg=sitk.ReadImage(datafn)
        img2=sitk.GetArrayFromImage(imgOrg)
 
        labelfilename='gt%d.nii'%ind  # provide a sample name of your filename of ground truth here
        labelfn=os.path.join(path,labelfilename)
        labelOrg=sitk.ReadImage(labelfn)
        labelimg=sitk.GetArrayFromImage(labelOrg)
        
        img=np.zeros([labelimg.shape[0],labelimg.shape[1],labelimg.shape[2],3])
        img[:,:,:,0]=img1;
        img[:,:,:,1]=img2;
        img[:,:,:,2]=labelimg;
        res=np.zeros(labelimg.shape)
        for i in range(0,img1.shape[0]):
            for j in range(0,img1.shape[1]):
                for k in range(0,img1.shape[2]):
                    arr=img[i,j,k,:]
                    e=mostFreqElement(arr)
                    res[i,j,k]=e
                    
#         b1_1=dice(labelimg,img1,1)
#         b2_1=dice(labelimg,img2,1)
#         if b1_1>b2_1:
#             res[img1==1]=1
#         else:
#             res[img2==1]=1
#             
#         b1_2=dice(labelimg,img1,2)
#         b2_2=dice(labelimg,img2,2)
#         if b1_2>b2_2:
#             res[img1==2]=2
#         else:
#             res[img2==2]=2
# 
#         b1_3=dice(labelimg,img1,3)
#         b2_3=dice(labelimg,img2,3)
#         if b1_3>b2_3:
#             res[img1==3]=3
#         else:
#             res[img2==3]=3

       
#         img2[img1==1]=1
#         img2[img1==2]=2
#         img2[img1==3]=3
        
#         labelfilename='Brain_1to1_CT_resampled_from4to1.hdr'  # provide a sample name of your filename of ground truth here
#         labelfn=os.path.join(path,labelfilename)
#         imgOrg=sitk.ReadImage(datafn)
#         img3=sitk.GetArrayFromImage(imgOrg)
        
#         img=(img1+img2+labelimg)/3
#         img=np.rint(img)

    
        
        volOut=sitk.GetImageFromArray(res)
        sitk.WriteImage(volOut,'preSub%d_MV_v1.nii'%ind)

if __name__ == '__main__':     
    main()
