# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 12:16:02 2016

This will be the main executing script of the system
This script is extended as the system grows

@author: Mats L. Richter
"""
import csv
import numpy as np
import os, os.path, sys
import MetaHandler

#INSERT IMPORT MODULES HERE
from Text_Score_Module import TextScore
from bow_pdf_test import BoW_Text_Module
from page_size_ratio_module import Page_Size_Module
from scanner_detect_module import ScannerDetect
from naive_scan_detector import Naive_Scan_Detector

#from ocr_bow_module import OCR_BoW_Module
#INSERT IMPORT OF NEURAL NETWORK HERE
from simple_neural_network import NN


def train(modules,neural_network,files,metafile):
    return

#prunes the filenames up to the point, the savefile saved the classifications last time, so the system can proceed
#instead of starting all over from the beginning
def prune(filenames, save_file):
    reader = csv.reader(save_file,delimiter=';')
    for row in reader:
        if row[0] in filenames:
            filenames.remove(row[0])
    return

#get all filenames
#the system currently assumes that all pdfs are stored in a single folder
#simply called 'files'
def get_files(path='./files'):
    filenames = list()
    for file in os.listdir(path):
        if file.endswith(".pdf"):
            filenames.append(file)
    return filenames
    
#builds a datavector from the given list of modules.
#all modules must feature a function 'get_function(filepointer, metapointer=None)'
#which will return a single float.
#
# @param modules:   the modules extracting the features
# @return           a R^len(modules) vector (numpy-array) the datavector has the same
#                   order as the modules
def get_data_vector(modules, filepointer, metapointer=None):
    data = list()
    for m in modules:
        try:
            #extract data-dimension from pdf
            data.append(m.get_function(filepointer,metapointer))
        except:
            #if erI mostlyror occures the value is 0.0
            data.append(0.0)
    #return as numpy array
    #return np.array(data)
    return data

#extracts features from a set of files and returns them on a list of lists
#it also saves the features to a csv file named features.csv
#
#@param filenames:   list of filenames
#@param classes:     list of classes from the classification.csv file
#@result:            numpy array of arrays to feed the NN
def extract_features(filenames,classes,metadata):
        
    c,m = get_classes(filenames,classes,metadata)
    
    feat_matrix = list()

    for f in range(len(filenames)):
        res=get_data_vector(modules,open(path+'/'+filenames[f],'rb'))
        res.append(c[f])
        res.append(filenames[f])
        feat_matrix.append(res) 
    
    with open("output.csv","w") as f:
        writer = csv.writer(f)
        writer.writerows(feat_matrix)
        
    return feat_matrix
    
#function to get the classes and metadata of the specified files
#
#@param filenames:    list of files
#@param classes:      list of classes from the classification.csv file
#@return:             list of classes (in binary) and metadata
def get_classes(filenames,classes,metadata):
    
    c = list()
    m = list()
    
    for f in filenames:
        
        #Encoding labels to make them numerical
        #There's the possibility to use categorical labels
        #by using a softmax activation algorithm on the nn
        #Could this be another way to induce a sort of bias 
        #on the different outputs by using different loss
        #functions??
        if classes[f.split('.')[0]] == True:
            c.append(1.)
        else:
            c.append(0.)
                    
        try:
            m.append(metadata[f.split('.')[0]])
        except:
            print("No metadata available for this file",f)
            
    return c,m
        
#loads the features from the csv file created during extraction
#
#@result:                 list of lists of features, list of corresponding classes and filenames
def load_data(features_file):
    
    with open(features_file, 'r') as f:
      reader = csv.reader(f)
      data = list(reader)
      
    num_features = len(data[0])-2
            
    features = [item[:num_features] for item in data]
    classes = [item[num_features] for item in data]
    filenames = [item[num_features+1] for item in data]
    
    return features, classes, filenames
    
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
    
    
    
#function to train modules if needed. Each module called should have a train function
def train_modules(modules,filenames,classes,metadata):
    #For now modules are pre-trained
    #We want a separate function for this
    for module in modules:
        module.train(filenames,classes,metadata)
    

def get_filepointer(path,filename):
    return open(path+'/'+filename,'rb')

def get_metapointer(path):
    return MetaHandler.get_whole_metadata(path)

def save_result(classes, save_file):
    """
    @input classes: the saved data as a list of 3-tuples
                    (filename, data_vector, classification)
    @return:        True if saves succesfully
    """
    for data in classes:
        save_string = data[0]+';'
        for d in data[1]:
            save_string = save_string+str(d)+';'
        save_string = save_string + data[2]
        save_file.write(save_string+"\n")
        save_file.flush()
    return True

#this function will initialize the neural network
# @parram input_dim:    number of modules for the input vector
def getNN(input_dim):
    network = NN()
    network.initializeNN(input_dim)
    return network

args = sys.argv
len_args = len(args)
training = False
extracting = False
#train mode
if '-t' in args:
    training = True
    TESTSIZE = 1000#should we use a percentage of the total data instead? Ex. 80%
    if len_args == 3:
        features_file = args[2]
    elif len_args == 4:
        features_file = args[3]
        
    if not os.path.isfile(features_file):
        print("Error: Features file doesn't exist.")
        exit();
    
if '-e' in args:
    extracting= True
    TESTSIZE = 1000#should we use a percentage of the total data instead? Ex. 80%     

#init filepointer for save-file here, the file will contain all classifications
#maybe csv-file, not yet decided
#create file if nonexistent
if(not os.path.isfile('classes.csv')):
    with open('classes.csv', 'w') as outcsv:
        writer = csv.writer(outcsv,delimiter=';')
        writer.writerow(["Filename", "NaiveScan", "Classification"])
#open file for pruning
save_file = open('classes.csv','r')

#the threshold for the neurral Network confidence
conf_thresh = 0.5

#initialize module
modules = list()
modules.append(TextScore())
modules.append(BoW_Text_Module())
modules.append(Page_Size_Module())
modules.append(ScannerDetect())
#modules.append(Naive_Scan_Detector())
#modules.append(OCR_BoW_Module())
#ADD MODULES HERE

#init neural network
network = getNN(len(modules))

#get filenames
path = './files'
filenames = get_files(path)
if training or extracting:
    metadata,classes = MetaHandler.get_classified_metadata("metadata.csv","classification.csv")
else:
    metadata = get_metapointer('metadata.csv')
#prune filenames if the system crashed
prune(filenames, save_file)
save_file.close()
#open for writing new data
save_file = open('classes.csv','a')

#EXTRACT FEATURES HERE
if(extracting):
    print("Extracting Features from the training set. This will take a while...")
    extract_features(filenames,classes,metadata)
    

#START TRAINING HERE
if(training):
    
    #train, filenames = MetaHandler.gen_train_test_split(filenames,TESTSIZE)
        
    features,classes,files = load_data(features_file)
        
    features = [[float(j) for j in i] for i in features]
    
    classes = [float(i) for i in classes]
    
    max_nor=max(map(lambda x: x[2], features))
    max_nor2=max(map(lambda x: x[1], features))
    
    min_nor=min(map(lambda x: x[2], features))
    min_nor2=min(map(lambda x: x[1], features))

#    for f in features: f[2] /= max_nor
#        
#    for f in features: f[1] /= max_nor2
    
    for f in features: (f[2] - min_nor)/(max_nor-min_nor)
        
    for f in features: (f[1] - min_nor2)/(max_nor2-min_nor2)
    
    features=np.array([np.array(xi) for xi in features])
    
    network.trainNN(features,np.array(classes))
    print("Training done!")



#classification
#batch size for the saving process
#batch_size = 10
#counter = 0
##this list will hold all classifications
#classes = list()
#for f in filenames:
#    counter += 1
#    print(str(counter)+'/'+str(len(filenames)))
#    if(counter%batch_size == 0 and not counter == 0 and batch_size != -1):
#        save_result(classes,save_file)
#        classes = list()
#    try:
#        fp = get_filepointer(path,f)
#        mp = metadata[f]
#    except:
#        print('Error opening file '+f)
#        continue
#    dat = get_data_vector(modules, fp,mp)
#    result = 0#network.predict(dat)
#
#    #interpret value of NN as confidence value
#    #the threshold serves as a adjustable bias for the classification
#    if result >= conf_thresh:
#        classes.append((f,dat,'True'))
#    else:
#        classes.append((f,dat,'False'))
#    fp.close()
#    #mp.close()
#        
#if(batch_size == -1):
#    save_result()
#save_file.close()

#ADD SOMETHING FOR PROCESSING RSUltS HERE