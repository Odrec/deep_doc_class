# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 18:58:33 2016

@author: odrec
"""
import sys, csv

#merges the features from two csv files created during extraction
#
#@param file1/file2:    files to merge
#@result:               new file (output_merged.csv) with features merged
def merge_features(file1,file2):
    
    data1, data2, filenames = open_files(file1, file2, 'm')
            
    num_features1 = len(data1[0])-2
    num_features2 = len(data2[0])-2
    
    features1 = [item[:num_features1] for item in data1]
    classes = [item[num_features1] for item in data1]
    
    features2 = [item[:num_features2] for item in data2]
                            
    features=[x + features2[i] for i, x in enumerate(features1)]
    
    data=[features[i]+[x] for i, x in enumerate(classes)]
    
    data=[data[i]+[x] for i, x in enumerate(filenames)]

    with open("output_merged.csv","w") as f:
        writer = csv.writer(f)
        writer.writerows(data)

    return True
    
#substitute an old feature with a new one
#
#@param orig_file:       the original file that has the old feature to substitute
#@param new_feat_file:   the file with the new feature to replace the old one
#@param feat_position:   the position of the old feature on the orig_file
#
#@result:               new file (output_sub.csv) with the new feature added
def substitute_feature(orig_file, new_feat_file, feat_position):
    
    orig_data, new_data = open_files(orig_file, new_feat_file, 's')
            
    orig_num_features = len(orig_data[0])-2
    
    fp = int(feat_position)
        
    if fp < orig_num_features:
                                
        for i, x in enumerate(orig_data): x[fp] = new_data[i][0]

        with open("output_sub.csv","w") as f:
            writer = csv.writer(f)
            writer.writerows(orig_data)
            
    else:
        print("ERROR: The position of the feature to substitute is incorrect")
        return False
        
    return True
    
#opens both files and checks if the data is correct
#
#@param file1/file2:    files to open and check
#@return:               the data from the files and the names of the files 
def open_files(file1, file2, origin):
    
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
            
            if origin == 'm':
                return data1, data2, filenames1
                
            if origin == 's':
                return data1, data2
            
        else:
            print("ERROR: Both files should have features for the same list of entries")
            return False
        
    else:
        print("ERROR: Both files to merge should contain features for the same amount of entries")
        return False
    
    
    
args = sys.argv

if len(args) == 3:
    merge_features(args[1],args[2])
else:
    
    if len(args) == 4:
        substitute_feature(args[1],args[2],args[3])
    else:
        print("You need to specify both files to merge as arguments, or two files and a position to substitute an old feature for a new one (the first file passed as argument should be the one with the old feature)")
