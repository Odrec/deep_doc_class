# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 15:10:43 2016

This code is used to separate the files marked as published (copyrighted)
in the classification.csv file from the non published (not copyrighted).

The copyrighted files are copied to the folder True/ while the non-copyrighted
are copied to the folder False/

Usage: python Copyfiles.py [path_to_files] [path_to classification.csv]

@author: Renato
"""

import csv, sys, os
import shutil
     
# @param files_path The filesystem path to the files
# @return returns some quantitative data to evaluate 
# the amount of files copied and missing
# --cft: counter for files that are marked as copyrighted in classification.csv
# --cff: counter for files that are marked as not copyrighted in classification.csv
# --other: counter for files that are not marked as copyrighted or not copyrighted (should be 0 always)
# --cct: counter for copied files that are marked as copyrighted
# --ccf: counter for copied files that are marked as not copyrighted
def copyFiles(files_path,classfile_path):
    
    cft=cff=other=cct=ccf=0
    
    with open(classfile_path) as classcsvfile: 
        classdata = csv.DictReader(classcsvfile, delimiter=';')
        
        if not os.path.exists('True'):
            os.mkdir('True')
        if not os.path.exists('False'):
            os.mkdir('False')
            
        for row in classdata:
            filename = row['document_id']
            fileclass = row['published']
            if fileclass == 'True':
                cft += 1
            else:
                if fileclass == 'False':
                    cff += 1
                else:
                    other += 1
                    
            for root, dirs, files in os.walk(files_path):
                for name in files:
                    if filename == os.path.splitext(os.path.basename(name))[0]:
                        
                        if fileclass == 'True':
                            cct += 1
                            shutil.copy2(os.path.join(root,name),'True/')
                        else:
                            ccf += 1
                            shutil.copy2(os.path.join(root,name),'False/')

    return cft,cff,cct,ccf,other
        

if len(sys.argv) == 3:
    files_path=sys.argv[1]
    classfile_path=sys.argv[2]
    
    cft,cff,cct,ccf,other=copyFiles(files_path,classfile_path)
    
    print("Files marked as True: ",cft)
    print("Files found and copied that were marked as True: ",cct)
    print("Files marked as False: ",cff)
    print("Files found and copied as that were marked as False: ",ccf)
    print("Other values: ",other)
else:
    print('Usage: python Copyfiles.py [path_to_files] [path_to classification.csv]')
        

