'''
Target: Decode one whole label map into three label maps which corresponds the all the organs
Created on Jan, 22th 2016
Author: Dong Nie 
'''



import SimpleITK as sitk

from multiprocessing import Pool
import os
import h5py
import numpy as np  
import scipy.io as scio

path='highresCTMR_29/'
def main():
    ids=[14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
    ids=[1,8]
    prename='gt'
    for id in ids:
        gtfn=os.path.join(path,prename+'%d.nii.gz'%id)
        outfn1=os.path.join(path,prename+'%d_1.nii.gz'%id)
        outfn2=os.path.join(path,prename+'%d_2.nii.gz'%id)
        outfn3=os.path.join(path,prename+'%d_3.nii.gz'%id)
        gtOrg=sitk.ReadImage(gtfn)
        gtMat=sitk.GetArrayFromImage(gtOrg)
        #gtMat=np.transpose(gtMat,(2,1,0))
        mat1=np.zeros((gtMat.shape[0],gtMat.shape[1],gtMat.shape[2]),dtype=np.int8)
        mat2=np.zeros((gtMat.shape[0],gtMat.shape[1],gtMat.shape[2]),dtype=np.int8)
        mat3=np.zeros((gtMat.shape[0],gtMat.shape[1],gtMat.shape[2]),dtype=np.int8)
        mat1[gtMat==1]=1
        mat2[gtMat==2]=1
        mat3[gtMat==3]=1
        gtVol1=sitk.GetImageFromArray(mat1)
        sitk.WriteImage(gtVol1,outfn1)
        gtVol2=sitk.GetImageFromArray(mat2)
        sitk.WriteImage(gtVol2,outfn2)
        gtVol3=sitk.GetImageFromArray(mat3)
        sitk.WriteImage(gtVol3,outfn3)  
#         prefn='preSub%d_as32_v12.nii'%id
#         preOrg=sitk.ReadImage(prefn)
#         preMat=sitk.GetArrayFromImage(preOrg)
#         preMat=np.transpose(preMat,(2,1,0))
#         preVol=sitk.GetImageFromArra(preMat)
#         sitk.WriteImage(preVol,prefn)
 
        
if __name__ == '__main__':     
    main()
