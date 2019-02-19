'''
by Dong Nie
05/03
target: preprocess the files. make directories, randomly cp some indexed files to other directories
'''

import numpy
import SimpleITK as sitk
import os
from doctest import SKIP
from shutil import copyfile
import random


class ScanFile(object):   
    def __init__(self,directory,prefix=None,postfix=None):  
        self.directory=directory  
        self.prefix=prefix  
        self.postfix=postfix  
          
    def scan_files(self):    
        files_list=[]    
            
        for dirpath,dirnames,filenames in os.walk(self.directory):   
            ''''' 
            dirpath is a string, the path to the directory.   
            dirnames is a list of the names of the subdirectories in dirpath (excluding '.' and '..'). 
            filenames is a list of the names of the non-directory files in dirpath. 
            '''  
            for special_file in filenames:    
                if self.postfix:    
                    special_file.endswith(self.postfix)    
                    files_list.append(os.path.join(dirpath,special_file))    
                elif self.prefix:    
                    special_file.startswith(self.prefix)  
                    files_list.append(os.path.join(dirpath,special_file))    
                else:    
                    files_list.append(os.path.join(dirpath,special_file))    
                                  
        return files_list    
      
    def scan_subdir(self):  
        subdir_list=[]  
        for dirpath,dirnames,files in os.walk(self.directory):  
            subdir_list.append(dirpath)  
        return subdir_list  
    

def main():
    path = '/shenlab/lab_stor/dongnie/brats2018/h5Data_sz240/'
    subpath = 'LGG'
    outfn = subpath + '*.h5'
    inputdir=path+subpath
    scan=ScanFile(path, postfix='.h5')  
    subfiles = scan.scan_subdir()  
    p_threshold = 0.2
    for subfile in subfiles:
        if subfile==path or subfile=='..':
            continue
        
        p = random.random()
        if p>p_threshold:
            continue
        
        print 'subfile is, ',subfile
        
        ss = subfile.split('/')
        print 'ss is, ',ss, 'and s7 is, ',ss[len(ss)-1]
        
        sdir = ss[len(ss)-1]
        
        sfile_moving = subfile.replace(subpath,"valH5")
#         sdir_fixed=path+sdir+'_fixed'
#         sdir_fused=path+sdir+'_fused'
#         os.mkdir(sdir_moving)
#         os.mkdir(sdir_fixed)
#         os.mkdir(sdir_fused)
        
#         copyfile(path+sdir+'/*moving*', sdir_moving)
#         copyfile(path+sdir+'/*fixed*', sdir_fixed)
#         copyfile(path+sdir+'/*fused*', sdir_fused)


        os.system('mv '+'%s'%subfile+' %s'%sfile_moving)
 
#         os.system('cp /home/dongnie/warehouse/pelvicSeg/newData/pelvic_0209/%s/*fixed*'%sdir+' %s'%sdir_fixed)
#         os.system('cp /home/dongnie/warehouse/pelvicSeg/newData/pelvic_0209/%s/*fused*'%sdir+' %s'%sdir_fused)
        
        
#         outfn=ss[7]+'.nii.gz'
#         
#         reader = sitk.ImageSeriesReader()
# 
#         dicom_names = reader.GetGDCMSeriesFileNames(subdir)
#         reader.SetFileNames(dicom_names)
#         
#         image = reader.Execute()
#         
#         size = image.GetSize()
#         print( "Image size:", size[0], size[1], size[2] )
#         
#         print( "Writing image:", outfn)
#         
#         sitk.WriteImage(image,outfn)


if __name__ == '__main__':     
    main()