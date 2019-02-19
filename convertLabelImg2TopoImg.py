'''
Target: Transpose (permute) the order of dimensions of a image  
Created on March, 5th, 2017
Author: Dong Nie 
'''



import SimpleITK as sitk

from multiprocessing import Pool
import os
import h5py
import numpy as np  
import scipy.io as scio
from PIL import Image
import scipy.misc as scmi
from scipy import ndimage as nd


path='/shenlab/lab_stor5/dongnie/pelvic/'
outPath='/shenlab/lab_stor5/dongnie/pelvic/'

d = 16
marginD = [d,d,d]
prefixPredictedFN = outPath+'topo_'
def main():
    #ids=[14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
    ids=range(1,41)
    rate=2.0/3
    
    for id in ids:
        gtfn=os.path.join(path,'img%d_label_nie_nocrop.nii.gz'%id)
#         outfn=os.path.join(path,'P%d/V2Rct_all.nii.gz'%id)
        gtOrg = sitk.ReadImage(gtfn)
        gtMat = sitk.GetArrayFromImage(gtOrg)
        [row,col,leng] = gtMat.shape
        
        matGTOut=np.zeros([row+2*marginD[0],col+2*marginD[1],leng+2*marginD[2]])
        #print 'matGTOut shape is ',matGTOut.shape
        matGTOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]] = gtMat
            
        if marginD[0]!=0:
            matGTOut[0:marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=gtMat[marginD[0]-1::-1,:,:] #reverse 0:marginD[0]
            matGTOut[row+marginD[0]:matGTOut.shape[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=gtMat[gtMat.shape[0]-1:row-marginD[0]-1:-1,:,:] #we'd better flip it along the 1st dimension
        if marginD[1]!=0:
            matGTOut[marginD[0]:row+marginD[0],0:marginD[1],marginD[2]:leng+marginD[2]]=gtMat[:,marginD[1]-1::-1,:] #we'd flip it along the 2nd dimension
            matGTOut[marginD[0]:row+marginD[0],col+marginD[1]:matGTOut.shape[1],marginD[2]:leng+marginD[2]]=gtMat[:,gtMat.shape[1]-1:col-marginD[1]-1:-1,:] #we'd flip it along the 2nd dimension
        if marginD[2]!=0:
            matGTOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],0:marginD[2]]=gtMat[:,:,marginD[2]-1::-1] #we'd better flip it along the 3rd dimension
            matGTOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]+leng:matGTOut.shape[2]]=gtMat[:,:,gtMat.shape[2]-1:leng-marginD[2]-1:-1]
        
        matProb0 =np.zeros([row,col,leng]) 
        matProb1 =np.zeros([row,col,leng]) 
        matProb2 =np.zeros([row,col,leng])
        matProb3 =np.zeros([row,col,leng])
        for i in range(0+marginD[0],row+marginD[0]):
            for j in range(0+marginD[1],col+marginD[1]):
                for k in range(0+marginD[2],leng+marginD[2]):
                    vol = matGTOut[i-marginD[0]:i+marginD[0],j-marginD[1]:j+marginD[1],k-marginD[2]:k+marginD[2]]
                    unique, counts = np.unique(vol, return_counts=True)
                    st = dict(zip(unique, counts))
                    sum = np.sum(counts)
                    if st.has_key(0):
                        matProb0[i-marginD[0],j-marginD[1],k-marginD[2]] = 1.0*st[0]/sum
                    if st.has_key(1):
                        matProb1[i-marginD[0],j-marginD[1],k-marginD[2]] = 1.0*st[1]/sum
                    if st.has_key(2):
                        matProb2[i-marginD[0],j-marginD[1],k-marginD[2]] = 1.0*st[2]/sum
                    if st.has_key(3):
                        matProb3[i-marginD[0],j-marginD[1],k-marginD[2]] = 1.0*st[3]/sum
                        
        
        volProb = sitk.GetImageFromArray(matProb0)
        sitk.WriteImage(volProb, prefixPredictedFN+'p0_sub{:02d}'.format(id)+'.nii.gz')
        
        volProb = sitk.GetImageFromArray(matProb1)
        sitk.WriteImage(volProb, prefixPredictedFN+'p1_sub{:02d}'.format(id)+'.nii.gz')

        volProb = sitk.GetImageFromArray(matProb2)
        sitk.WriteImage(volProb, prefixPredictedFN+'p2_sub{:02d}'.format(id)+'.nii.gz')        

        volProb = sitk.GetImageFromArray(matProb3)
        sitk.WriteImage(volProb, prefixPredictedFN+'p3_sub{:02d}'.format(id)+'.nii.gz')        
                        
#         print 'mat shape, ', gtMat.shape
#         for s in range(1,gtMat.shape[0]):
#             sliceMat=gtMat[s-1,:,:]
#             sliceMatScale = nd.interpolation.zoom(sliceMat, zoom=rate)
#             scmi.imsave('p%d_'%id+'s%d.png'%s, sliceMat)
        #gtMat=np.transpose(gtMat,(2,1,0))
#         gtVol=sitk.GetImageFromArray(gtMat)
#         sitk.WriteImage(gtVol,outfn)
#         
#         prefn='preSub%d_as32_v12.nii'%id
#         preOrg=sitk.ReadImage(prefn)
#         preMat=sitk.GetArrayFromImage(preOrg)
#         preMat=np.transpose(preMat,(2,1,0))
#         preVol=sitk.GetImageFromArra(preMat)
#         sitk.WriteImage(preVol,prefn)
 
        
if __name__ == '__main__':     
    main()
