'''
Load npy format data, and save as nii.gz format data
'''
import numpy as np
import SimpleITK as sitk

ids=[1,9,10,11]

for i in range(0,len(ids)):
	id=ids[i]
	filename='preSub%d_as32.npy'%id
	mat=np.load(filename)
	outfn=filename[:-4]
	outfn=outfn+'.nii.gz'
	volOut=sitk.GetImageFromArray(mat)
	sitk.WriteImage(volOut,outfn)
