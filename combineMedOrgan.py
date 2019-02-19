    
'''
Target: combine for kinds of medical images (different organ images), to form a one-whole segmentation image
Created on Jan. 21, 2017
Author: Dong Nie 
'''



import SimpleITK as sitk

from multiprocessing import Pool
import os
import h5py
import numpy as np  
from imgUtils import dice

def main():
    path='./highresCTMR_29/'
    saveto='./highresCTMR_29/'
    #ids=[1,2,3,4,6,7,8,10,11,12,13]
    ids=[14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
    for ind in ids:
        print 'come to, ',ind
        datafilename='p%d/ct_bla.mhd'%ind #provide a sample name of your filename of data here
        datafn=os.path.join(path,datafilename)
        imgOrg=sitk.ReadImage(datafn)
        img_bla=sitk.GetArrayFromImage(imgOrg)
        
        
        labelfilename='p%d/ct_pro.mhd'%ind  # provide a sample name of your filename of ground truth here
        labelfn=os.path.join(path,labelfilename)
        imgOrg=sitk.ReadImage(labelfn)
        img_pro=sitk.GetArrayFromImage(imgOrg)
 
        labelfilename='p%d/ct_rec.mhd'%ind  # provide a sample name of your filename of ground truth here
        labelfn=os.path.join(path,labelfilename)
        labelOrg=sitk.ReadImage(labelfn)
        img_rec=sitk.GetArrayFromImage(labelOrg)
        
        img=np.zeros([img_rec.shape[0],img_rec.shape[1],img_rec.shape[2]])
        img[img_bla==255]=1
        img[img_pro==255]=2
        img[img_rec==255]=3
        #img[img==255]=1
        img=img.astype(int8)
        
        volOut=sitk.GetImageFromArray(img)
        sitk.WriteImage(volOut,saveto+'p%d/ct_seg.nii'%ind)
        print 'finish, ', ind
if __name__ == '__main__':     
    main()
