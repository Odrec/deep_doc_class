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
            #if error occures the value is 0.0
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

    print('Training Neural Network...')
    for f in range(len(filenames)):
        #nn_train.append(get_data_vector(modules,train[t],m[t])) #No metadata for now (Index out of range error)
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
            
        return (c,m)
        
#loads the features from the csv file created during extraction
#
#@param num_features:    quantity of features
#@result:                 list of lists of features, list of corresponding classes and filenames
def load_data():
    
    with open('output.csv', 'r') as f:
      reader = csv.reader(f)
      data = list(reader)
      
    num_features = len(data[0])-2
            
    features = [item[:num_features] for item in data]
    classes = [item[num_features] for item in data]
    filenames = [item[num_features+1] for item in data]
    
    return features, classes, filenames
    
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
training = False
extraining = False
#train mode
if '-t' in args:
    training = True
    TESTSIZE = 1000#should we use a percentage of the total data instead? Ex. 80%
    
if '-et' in args:
    training = True
    extraining = True
    TESTSIZE = 1000#should we use a percentage of the total data instead? Ex. 80%    


#init filepointer for save-file here, the file will contain all classifications
#maybe csv-file, not yet decided
#create file if nonexistent
if(not os.path.isfile('classes.csv')):
    with open('classes.csv', 'w') as outcsv:
        writer = csv.writer(outcsv,delimiter=';')
        writer.writerow(["Filename", "TextScore","BoWTextScore","Pagesize(KBytes)","ScannerDetect", "Classification"])
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
#modules.append(OCR_BoW_Module())
#ADD MODULES HERE

#init neural network
network = getNN(len(modules))

#get filenames
path = './files'
filenames = get_files(path)
if training:
    metadata,classes = MetaHandler.get_classified_metadata("metadata.csv","classification.csv")
else:
    metadata = get_metapointer('metadata.csv')
#prune filenames if the system crashed
prune(filenames, save_file)
save_file.close()
#open for writing new data
save_file = open('classes.csv','a')

#START TRAINING HERE
if(training):
    
    train, filenames = MetaHandler.gen_train_test_split(filenames,TESTSIZE)
    
    if(extraining):
        #print("Training Features...")
        print("Extracting Features from the training set. This will take a while...")
        extract_features(train,classes,metadata)
        
    features,classes,files = load_data()
        
    features = [[float(j) for j in i] for i in features]
    
    classes = [float(i) for i in classes]
    
    nor=max(map(lambda x: x[2], features))
    nor2=max(map(lambda x: x[1], features))

    for f in features: f[2] /= nor
        
    for f in features: f[1] /= nor2  
    
    features=np.array([np.array(xi) for xi in features])
    
    network.trainNN(features,np.array(classes))
    print("Training done!")



#classification
#batch size for the saving process
batch_size = 10
counter = 0
#this list will hold all classifications
classes = list()
for f in filenames:
    counter += 1
    print(str(counter)+'/'+str(len(filenames)))
    if(counter%batch_size == 0 and not counter == 0 and batch_size != -1):
        save_result(classes,save_file)
        classes = list()
    try:
        fp = get_filepointer(path,f)
        mp = metadata[f]
    except:
        print('Error opening file '+f)
        continue
    dat = get_data_vector(modules, fp,mp)
    result = 0#network.predict(dat)

    #interpret value of NN as confidence value
    #the threshold serves as a adjustable bias for the classification
    if result >= conf_thresh:
        classes.append((f,dat,'True'))
    else:
        classes.append((f,dat,'False'))
    fp.close()
    #mp.close()
        
if(batch_size == -1):
    save_result()
save_file.close()

#ADD SOMETHING FOR PROCESSING RSUltS HERE