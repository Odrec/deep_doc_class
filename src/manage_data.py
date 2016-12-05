# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 18:58:33 2016

@author: odrec
"""
import sys, csv, random, os, os.path

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
    # if(len(orig_pos_list)<=len(orig_data[0]) and len(new_pos_list)<=len(new_data[0]) and len(new_pos_list)==len(orig_pos_list)):
                        
        for i, x in enumerate(orig_data):
            x[fp] = new_data[i][0]
            # for op,np in zip(orig_pos_list,new_pos_list):
            #     x[op] = new_data[i][np]



        with open("output_sub.csv","w") as f:
            writer = csv.writer(f)
            writer.writerows(orig_data)
            
    else:
        # print("ERROR: The list are not correct")
        print("ERROR: The position of the feature to substitute is incorrect")
        return False
        
    return True
    
#detach a feature from an output file
#
#@param orig_file:       the original file that has the feature to detach
#@param feat_position:   the position of the feature to detach on the orig_file
#
#@result:               new file (output_det.csv) with the detached feature
def detach_feature(orig_file, feat_position):
    
    data = open_files(orig_file, None, 'o')
            
    orig_num_features = len(data[0])-2
    
    fp = int(feat_position)
            
    if fp < orig_num_features:
        
        class_file = [item[orig_num_features:orig_num_features+2] for item in data]
        
        new_data = list()
                                
        new_data = [x[fp] for x in data]
        
        new_data = [[new_data[i]] + x for i, x in enumerate(class_file)]

        with open("output_det.csv","w") as f:
            writer = csv.writer(f)
            writer.writerows(new_data)
            
    else:
        print("ERROR: The position of the feature to detach is incorrect")
        return False
        
    return True
    
    
#create a new output file with the subset of features for the same amount of 
#positive and false examples
#
#@param orig_file:       the original output file
#@param num_files:       the number of files to be used for each class. If not given it will take
#   the lowest number of files from both classes and use that one
#
#@result:               new file (output_subset.csv) with the sample features
def subset_features(orig_file, num_files=0):

    data = open_files(orig_file, None, 'o')
      
    ld = len(data[0])
    
    data_true = list()
    data_false = list()
    data_new = list()
      
    if num_files == 0:
        for x in data: 
            if x[ld-2] == '1.0': data_true.append(x)
            elif x[ld-2] == '0.0': data_false.append(x)
            
        num_files = min(len(data_true), len(data_false))
        
    rst = random.sample(data_true, num_files)
    rsf = random.sample(data_false, num_files) 
    
    data_new = rst + rsf
    
    random.shuffle(data_new)

    with open("output_subset.csv","w") as f:
        writer = csv.writer(f)
        writer.writerows(data_new)

    return True

#preapres the training data. Checks if the file exists and 
#creates a new csv with all the existing files on a particular path
#
#@param class_file:      the classification.csv file
#@param files_path:      the path to the actual files
#
#@result:                creates a training_data.csv file with the available training data on the path  
def prepare_training_data(class_file='classification.csv', files_path='./files'):
    
    with open(class_file) as cf:
        class_data = csv.DictReader(cf, delimiter=';')
        data_list = list(class_data)
    
    files_names = get_files(files_path)
    
    final_data_list = list()
    
    for d in data_list:
        for f in files_names:
            file_name = os.path.splitext(f)[0]
            sublist = []
            if file_name == d['document_id']:
                if d['published'] == 'False':
                    sublist.append(0.0)
                elif d['published'] == 'True':
                    sublist.append(1.0)
                sublist.append(file_name)
                final_data_list.append(sublist)
                break
            
    with open("training_data.csv","w") as f:
        writer = csv.writer(f)
        writer.writerows(final_data_list)
            
#get all filenames
def get_files(path):
    filenames = list()
    for file in os.listdir(path):
        if file.endswith(".pdf"):
            filenames.append(file)
    return filenames

#opens both files and checks if the data is correct
#
#@param file1/file2:    files to open and check
#@param original        depending on which function calls it the return data is different
#@return:               the data from the files and the names of the files 
def open_files(file1, file2, origin):
    
    with open(file1, 'r') as f:
      reader1 = csv.reader(f)
      data1 = list(reader1)
      
    if origin == 'o':
        return data1
      
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
len_args = len(args)
hlp = False

if '-h' in args:
    
    hlp = True
    
elif '-t' in args:
    
    if len_args == 2:
        prepare_training_data()
    elif len_args == 3:
        prepare_training_data(args[2])
    elif len_args == 4 and os.path.isdir(args[3]):
        prepare_training_data(args[2], args[3])
    
elif os.path.isfile(args[2]):

    if '-d' in args and len_args == 4 and args[3].isdigit():
        
        detach_feature(args[2], args[3])        
        
    elif '-s' in args:
        
        if len_args == 3:
            subset_features(args[2])
        elif args[3].isdigit():
            subset_features(args[2], args[3])
        else: fail = True
        
    elif '-m' in args and len_args == 4 and os.path.isfile(args[2]) and os.path.isfile(args[3]):
        
        merge_features(args[2], args[3])
                
    elif '-c' in args and len_args == 5 and os.path.isfile(args[2]) and os.path.isfile(args[3]) and args[4].isdigit():
                
        substitute_feature(args[2], args[3], args[4])
            
    else: hlp = True

if hlp:
    print("\n")
    print("Usage:")
    print("\n")
    print("-h")
    print("\n")
    print("Prints this help")
    print("\n")
    print("-m [path_to_output_file_1] [path_to_output_file_2]")
    print("\n")
    print("This will merge the features on both files into a new output_merged.csv")
    print("\n")
    print("-c [path_to_output_file_with_old_feature] [path_to_output_file_with_new_feature] [position_of_old_feature_on_first_output_file]")
    print("\n")
    print("This will substitute an old feature on a determined position on the first file with a new feature on the first position of the second file and create a new ouput_sub.csv file")
    print("\n")
    print("-s [path_to_output_file] [[number of files to subset]]")
    print("\n")
    print("This will choose an equal amount of random True files and False files and create a new output_sample.csv file")
    print("\n")
    print("-d [path_to_output_file] [position_of_feature_to_detach]")
    print("\n")
    print("This will detach one feature from a file and create a new output_det.csv file with that feature")
    print("\n")
    print("-t [[path_to_calssifications_file]] [[path_to_training_files]]")
    print("\n")
    print("This will create a new csv file (training_data.csv) where the training data for the files in the folder can be found")
    print("\n")