# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 18:58:33 2016

@author: odrec
"""
import sys, csv

#merges the features from two csv files created during extraction
#
#@param file1/file2:    files to merge
#@result:               new file with features merged
def merge_features(file1,file2):
    
    with open(file1, 'r') as f:
      reader1 = csv.reader(f)
      data1 = list(reader1)
      
    with open(file2, 'r') as f:
      reader2 = csv.reader(f)
      data2 = list(reader2)
      
    num_files1 = len(data1)
    num_files2 = len(data2)
    
    if num_files1 == num_files2:
        
        len1 = len(data1[0])
        len2 = len(data2[0])
        
        data1.sort(key=lambda x: x[len1-1])
        data2.sort(key=lambda x: x[len2-1])
    
        filenames1 = [item[len1-1] for item in data1]
        filenames2 = [item[len2-1] for item in data2]
    
        if filenames1 == filenames2:
            
            num_features1 = len1-2
            num_features2 = len2-2
            
            features1 = [item[:num_features1] for item in data1]
            classes = [item[num_features1] for item in data1]
            
            features2 = [item[:num_features2] for item in data2]
                                    
            features=[x + features2[i] for i, x in enumerate(features1)]
            
            data=[features[i]+[x] for i, x in enumerate(classes)]
            
            data=[data[i]+[x] for i, x in enumerate(filenames1)]

            with open("output_new.csv","w") as f:
                writer = csv.writer(f)
                writer.writerows(data)

        else:
            print("ERROR: Both files should have features for the same list of entries")
            return False
        
    else:
        print("ERROR: Both files to merge should contain features for the same amount of entries")
        return False
        
    return True
    
    
    
args = sys.argv

if len(args) == 3:
    merge_features(args[1],args[2])
else:
    print("You need to specify both files to merge as arguments.")
