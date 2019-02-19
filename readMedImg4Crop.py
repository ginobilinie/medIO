    
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

def cropCubic(idx,subName,path_subjects,volsz,stride,destres,samplespersubject,saveto):
    print subName
    cov=np.array([[volsz[0]/2,0,0],[0,volsz[1]/2,0],[0,0,volsz[2]/2]])
    X=np.zeros((samplespersubject,1,volsz[2],volsz[1],volsz[0]),dtype='float32')#data
    y=np.zeros((samplespersubject,volsz[2],volsz[1],volsz[0]),dtype='uint8')#labels
    
    subject=os.path.join(path_subjects,subName,subName+'.nii.gz')
    label=os.path.join(path_subjects,subName,'GT.nii.gz')
    img=sitk.ReadImage(subject)
    normfilter=sitk.NormalizeImageFilter()
    img=normfilter.Execute(img)
    imglabel=sitk.ReadImage(label)
    factor = np.asarray(img.GetSpacing()) / destres
    
    factorSize = np.asarray(img.GetSize() * factor, dtype=float)
    
    newSize = np.max([factorSize, volsz], axis=0)
    
    newSize = newSize.astype(dtype=int)
    
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img)
    resampler.SetOutputSpacing(destres)
    resampler.SetSize(newSize)
    resampler.SetInterpolator(sitk.sitkLinear)
    imgResampled = resampler.Execute(img)
    
    resampler.SetReferenceImage(imglabel)
    resampler.SetOutputSpacing(destres)
    resampler.SetSize(newSize)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)            
    labelResampled = resampler.Execute(imglabel)
    
    segnp=sitk.GetArrayFromImage(labelResampled)
    listorgan=np.where(segnp>0)
    zmin=np.min(listorgan[0])
    zmax=np.max(listorgan[0])    
    imgCentroid = np.asarray(newSize, dtype=float) / 2.0
    imgCentroid[2]=zmin+(zmax-zmin)/2.0#just for test
    regionExtractor = sitk.RegionOfInterestImageFilter()
    regionExtractor.SetSize(list(volsz.astype(dtype=int)))
    for sampleid in xrange(samplespersubject):        
        print subName+' sample-> ',sampleid
        pts= np.random.multivariate_normal(imgCentroid, cov)#use centroid as mean
        imgStartPx = (pts - volsz / 2.0).astype(dtype=int)
        
        #print 'px start ',imgStartPx
        
        regionExtractor.SetIndex(list(imgStartPx))
        
        imgResampledCropped = regionExtractor.Execute(imgResampled)
        labelResampledCropped = regionExtractor.Execute(labelResampled)
        ctnp=sitk.GetArrayFromImage(imgResampledCropped)
        labelnp=sitk.GetArrayFromImage(labelResampledCropped)
        
        X[sampleid,0,...]=ctnp
        y[sampleid,...]=labelnp
        
    #shuffle  
    print subName+' shuffling'     
    idx_rnd=np.random.choice(X.shape[0], X.shape[0], replace=False)
    X=X[idx_rnd]
    y=y[idx_rnd]
    train_filename = os.path.join(saveto, 'train{}.h5'.format(idx))
    comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
    print subName+' saving hdf5...'
    with h5py.File(train_filename, 'w') as f:
        f.create_dataset('data', data=X, **comp_kwargs)
        f.create_dataset('label', data=y.astype(np.float32), **comp_kwargs)        
    print "subject  {0} finishedm type:  {1}!".format(subName, X.dtype)   
    
def cropCubicv1(idx,subName,path_subjects,volsz,stride,destres,samplespersubject,saveto):
    print subName
    cov=np.array([[volsz[0]/2,0,0],[0,volsz[1]/2,0],[0,0,volsz[2]/2]])
    
    patch_sz=[volsz[2],volsz[1],volsz[0]]
    #stride=[patch_sz[0]/2,patch_sz[1]/2,patch_sz[2]/2]
    
    subject=os.path.join(path_subjects,subName,subName+'.nii.gz')
    label=os.path.join(path_subjects,subName,'GT.nii.gz')
    body=os.path.join(path_subjects,subName,'CONTOUR.nii.gz')
    imgtmp=sitk.ReadImage(subject)
    nptmp=sitk.GetArrayFromImage(imgtmp)
    nptmp[np.where(nptmp>3000)]=3000#clamp it
    img=sitk.GetImageFromArray(nptmp)
    img.SetSpacing(imgtmp.GetSpacing())
    img.SetOrigin(imgtmp.GetOrigin())
    img.SetDirection(imgtmp.GetDirection())
    normfilter=sitk.NormalizeImageFilter()
    img=normfilter.Execute(img)
    imglabel=sitk.ReadImage(label)
    imgbody=sitk.ReadImage(body)
    factor = np.asarray(img.GetSpacing()) / destres
    
    factorSize = np.asarray(img.GetSize() * factor, dtype=float)
    
    newSize = np.max([factorSize, volsz], axis=0)
    
    newSize = newSize.astype(dtype=int)
    
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img)
    resampler.SetOutputSpacing(destres)
    resampler.SetSize(newSize)
    resampler.SetInterpolator(sitk.sitkLinear)
    imgResampled = resampler.Execute(img)
    resampler.SetReferenceImage(imglabel)
    resampler.SetOutputSpacing(destres)
    resampler.SetSize(newSize)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)            
    labelResampled = resampler.Execute(imglabel)
    bodyResampled=resampler.Execute(imgbody)
    
    segnp=sitk.GetArrayFromImage(labelResampled)
    
    
    regionExtractor = sitk.RegionOfInterestImageFilter()
    regionExtractor.SetSize(list(volsz.astype(dtype=int)))
    shape=segnp.shape
    
    
    #(N-F)/S +1
    nptsz=(shape[0]-patch_sz[0])/stride[0]+1
    nptsr=(shape[1]-patch_sz[1])/stride[1]+1
    nptsc=(shape[2]-patch_sz[2])/stride[2]+1
    
    totalpts=nptsc*nptsr*nptsz
    
    X=np.zeros((samplespersubject+totalpts,1,volsz[2],volsz[1],volsz[0]),dtype='float32')#data
    y=np.zeros((samplespersubject+totalpts,volsz[2],volsz[1],volsz[0]),dtype='uint8')#labels
    idpatch=0
    list_remove=[]
    for z in range(patch_sz[0]/2,shape[0]-patch_sz[0]/2+1,stride[0]):
        print z
        for r in range(patch_sz[1]/2,shape[1]-patch_sz[1]/2+1,stride[1]):
            for c in range(patch_sz[2]/2,shape[2]-patch_sz[2]/2+1,stride[2]):
                imgCentroid = np.asarray(([c,r,z]), dtype=float)#x-y-z
                imgStartPx = (imgCentroid - volsz / 2.0).astype(dtype=int)
                if bodyResampled[[c,r,z]]==0:
                    list_remove.append(idpatch)
                regionExtractor.SetIndex(list(imgStartPx))
                imgResampledCropped = regionExtractor.Execute(imgResampled)
                labelResampledCropped = regionExtractor.Execute(labelResampled)
                               
                patchct=sitk.GetArrayFromImage(imgResampledCropped)
                patchgt=sitk.GetArrayFromImage(labelResampledCropped)
                X[idpatch,0,...]=patchct
                y[idpatch,...]=patchgt
                idpatch+=1

    listorgan=np.where(segnp>0)
    zmin=np.min(listorgan[0])
    zmax=np.max(listorgan[0])    
    imgCentroid = np.asarray(newSize, dtype=float) / 2.0
    imgCentroid[2]=zmin+(zmax-zmin)/2.0#just for test
    for sampleid in xrange(samplespersubject):        
        print subName+' sample-> ',sampleid
        pts= np.random.multivariate_normal(imgCentroid, cov)#use centroid as mean
        imgStartPx = (pts - volsz / 2.0).astype(dtype=int)
        print imgStartPx
        #print 'px start ',imgStartPx
        
        regionExtractor.SetIndex(list(imgStartPx))
        
        imgResampledCropped = regionExtractor.Execute(imgResampled)
        labelResampledCropped = regionExtractor.Execute(labelResampled)
        ctnp=sitk.GetArrayFromImage(imgResampledCropped)
        labelnp=sitk.GetArrayFromImage(labelResampledCropped)
        
#         idbg=np.where(labelnp==0)
#         ideso=np.where(labelnp==1)
#         idheart=np.where(labelnp==2)
#         idtrach=np.where(labelnp==3)
#         idaorta=np.where(labelnp==4)
#         
#         total_pix=np.prod(labelnp.shape)
#         total_pix=total_pix.astype(np.float32)
#         print 'bg perc ',len(idbg[0])/total_pix
#         print 'eso perc ',len(ideso[0])/total_pix
#         print 'heart perc ',len(idheart[0])/total_pix
#         print 'trach perc ',len(idtrach[0])/total_pix
#         print 'aorta perc ',len(idaorta[0])/total_pix
        X[sampleid+totalpts,0,...]=ctnp
        y[sampleid+totalpts,...]=labelnp
        #remove rows 
    X=np.delete(X,list_remove,axis=0)
    y=np.delete(y,list_remove,axis=0)
    totalpts=X.shape[0]+samplespersubject
    print subName+' total patches ',totalpts   
    #shuffle  
    print subName+' shuffling'     
    idx_rnd=np.random.choice(X.shape[0], X.shape[0], replace=False)
    X=X[idx_rnd]
    y=y[idx_rnd]
    train_filename = os.path.join(saveto, 'train{}.h5'.format(idx))
    comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
    print subName+' saving hdf5...'
    with h5py.File(train_filename, 'w') as f:
        f.create_dataset('data', data=X, **comp_kwargs)
        f.create_dataset('label', data=y.astype(np.float32), **comp_kwargs)        
    print "subject  {0} finished type:  {1}!".format(subName, X.dtype)       
    
def resample(path_subjects,volsz,destres,samplespersubject,saveto):
    _, subjects, _ = os.walk(path_subjects).next()#every folder is a subject
    print subjects   
    
    dirname =saveto
    if not os.path.exists(dirname):
        os.makedirs(dirname) 
    
    
    h5names=[os.path.join(dirname,'train{0}.h5\n'.format(idx)) for idx,_ in enumerate(subjects)]
    f =open(os.path.join(dirname, 'train.txt'), 'w')
    f.writelines(h5names)
       
    pool = Pool(processes=2)
    
    
    for idx,subName in enumerate(subjects):
        pool.apply_async(cropCubicv1,args=(idx,subName,path_subjects,volsz,stride,destres,samplespersubject,dirname))
        #cropCubic(idx,subName,path_subjects,volsz,destres,samplespersubject,dirname)
    
    pool.close()
    pool.join() 
    print 'finished'
    
    
def main():
    path_subjects='/home/dongnie/warehouse/CT_subjects/train_set/'
    saveto='/home/dongnie/warehouse/prostate'
    destres=np.array([1.0,1.0,2.0])
    volsz=np.array([160,160,64])
    samplespersubject=32
    resample(path_subjects, volsz, destres,samplespersubject,saveto)
    
if __name__ == '__main__':     
    main()
