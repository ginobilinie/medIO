import numpy as np
import h5py
import SimpleITK as sitk

filename = 'train32Sample_1.h5'

with h5py.File(filename,'r') as hf:
    dataT1 = hf.get('dataT1')
    dataT2 = hf.get('dataT2')
    dataSeg = hf.get('dataSeg')
    dT1 = dataT1[0,0,:,:,:]
    dT2 = dataT2[0,0,:,:,:]
    dSeg = dataSeg[0,0,:,:,:]
    print dataT1.shape
    print dataT2.shape
    print dataSeg.shape
    print dT1.shape
    print type(dT1)
    print 'haha'
    print type(dT1[0,0,0])
    print 'voxel is ',dT1[15,15,15]
    dT1 = dT1.astype(np.float64)
    dT2 = dT2.astype(np.float64)
    volT1 = sitk.GetImageFromArray(dT1)
    print 'haha 1'
    volT2 = sitk.GetImageFromArray(dT2)
    sitk.WriteImage(volT1,'patchT1.nii.gz')
    sitk.WriteImage(volT2,'patchT2.nii.gz')

    volSeg = sitk.GetImageFromArray(dSeg)
    sitk.WriteImage(volSeg,'patchSeg.nii.gz')