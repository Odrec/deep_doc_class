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
from multiprocessing import Pool
import MetaHandler

#INSERT IMPORT MODULES HERE
from Text_Score_Module import TextScore
from bow_pdf_test import BoW_Text_Module
from page_size_ratio_module import Page_Size_Module
from scanner_detect_module import ScannerDetect
from bow_metadata import Bow_Metadata
from orientation_detector import page_orientation_module
from Negative_BoW import Negative_BoW_Text_Module
from resolution_module import Resolution_Module

#from ocr_bow_module import OCR_BoW_Module

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
    
#builds a datavector from the given list of modules.
#all modules must feature a function 'get_function(filepointer, metapointer=None)'
#which will return a single float.
#
# @param modules:   the modules extracting the features
# @return           a R^len(modules) vector (numpy-array) the datavector has the same
#                   order as the modules
def get_data_vector(file_data, metapointer=None):
    
    filepointer = open('./files/'+file_data[1]+'.pdf','rb')
    feature_data = list()
    
    for m in modules:
        try:
            #extract data-dimension from pdf
            feature_data.append(m.get_function(filepointer,metapointer))
        except:
            #if error occures the value is nan
            feature_data.append(np.nan)

    return [file_data,feature_data]

#extracts features from a set of files and returns them on a list of lists
#it also saves the features to a csv file named features.csv
#
#@param filenames:   list of filenames
#@param classes:     list of classes from the classification.csv file
#@result:            numpy array of arrays to feed the NN
#def extract_features(filenames,classes,metadata):
#        
#    c,m = get_classes(filenames,classes,metadata)
#    
#    feat_matrix = list()
#
#    for f in range(len(filenames)):
#        res=get_data_vector(modules,open(path+'/'+filenames[f],'rb'))
#        res.append(c[f])
#        res.append(filenames[f])
#        feat_matrix.append(res) 
#    
#    with open("output.csv","w") as f:
#        writer = csv.writer(f)
#        writer.writerows(feat_matrix)
#        
#    return feat_matrix
    
def extract_features(data, metadata=None):
        
    #c,m = get_classes(filenames,classes,metadata)
    
    feat_matrix = list()
    
    if p == -1:
        pool = Pool()
    else:
        pool = Pool(p)
    
    res = pool.map(get_data_vector, data)
    
    features = list()
    file_data = list()
    
    for item in res:
        features.append(item[1])
        file_data.append(item[0]) 
    
    for f, r in enumerate(features):
        r.append(file_data[f][0])
        r.append(file_data[f][1])
        feat_matrix.append(r) 
    
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
#@result:   list of lists of features, list of corresponding classes and filenames
def load_data(features_file):
    
    with open(features_file, 'r') as f:
      reader = csv.reader(f)
      data = list(reader)
      
    num_features = len(data[0])-2
            
    features = [item[:num_features] for item in data]

    features = generate_error_features(features)     
    
    classes = [item[num_features] for item in data]
    filenames = [item[num_features+1] for item in data]
    
    return features, classes, filenames
    
#generates the error features and replaces the nan values
#
#@result:   list of features with the error features included
def generate_error_features(features):
    
    error_feature = [0.0] * len(features)
    
    for i, x in enumerate(features):
        for j, e in enumerate(x):
            if e == 'nan':
                error_feature[i] = 1.0
                x[j] = 1.0                
    
    features = [x + [error_feature[i]] for i, x in enumerate(features)]
    return features
    
    
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
def getNN(input_dim, hidden_dim):
    network = NN()
    network.initializeNN(input_dim, hidden_dim)
    return network
    
#plots several efficiency measurements
def plot(nodes, epochs, bias, features, classes, ):
    
    import matplotlib.pyplot as plt
    
    import time
    
    fig = 0
        
    if nodes:
        num_nodes = [10, 20, 50, 100, 200, 500]
        ll = len(num_nodes)
        ac = [None]*ll
        f1 = [None]*ll
        pr = [None]*ll
        rc = [None]*ll
        ex = np.empty([ll,4])
        tm = [None]*ll
        
        for i, x in enumerate(num_nodes):
            network = getNN(len(features[0]), x)
            start_time = time.time()
            ac[i], f1[i], pr[i], rc[i], ex[i][0], ex[i][1], ex[i][2], ex[1][3] = network.trainNN(features, np.array(classes), 100, .5)
            tm[i] = time.time() - start_time
            
        fig+=1
            
        plt.figure(fig)
        plt.plot(num_nodes, ac)
        plt.ylabel('accuracy')
        plt.xlabel('number of nodes')
        plt.axis([ 0, num_nodes[ll-1]+50, min(ac), max(ac) ])
        
        fig+=1
        
        plt.figure(fig)
        plt.plot(num_nodes, f1)
        plt.ylabel('f1')
        plt.xlabel('number of nodes')
        plt.axis([ 0, num_nodes[ll-1]+50, min(f1), max(f1) ])
        
        fig+=1
        
        plt.figure(fig)
        plt.plot(num_nodes, pr)
        plt.ylabel('precision')
        plt.xlabel('number of nodes')
        plt.axis([ 0, num_nodes[ll-1]+50, min(pr), max(pr) ])
        
        fig+=1        
        
        plt.figure(fig)
        plt.plot(num_nodes, rc)
        plt.ylabel('recall')
        plt.xlabel('number of nodes')
        plt.axis([ 0, num_nodes[ll-1]+50, min(rc), max(rc) ])
        
        fig+=1        
        
        plt.figure(fig)
        lineObjects = plt.plot(num_nodes, ex)
        plt.legend(lineObjects, ('tn', 'tp', 'fn', 'fp'))
        plt.ylabel('examples')
        plt.xlabel('number of nodes')
        #plt.axis([ 0, num_nodes[ll-1]+50, 0, 300 ])
        
        fig+=1        
        
        plt.figure(fig)
        plt.plot(num_nodes, tm)
        plt.ylabel('time in seconds')
        plt.xlabel('number of nodes')
        plt.axis([ 0, num_nodes[ll-1]+100, min(tm), max(tm) ])
                
    if epochs:
        num_epochs = [50, 100, 200, 500, 1000, 2000]
        ll = len(num_epochs)
        ac = [None]*ll
        f1 = [None]*ll
        pr = [None]*ll
        rc = [None]*ll
        ex = np.empty([ll,4])
        tm = [None]*ll
                
        for i, x in enumerate(num_epochs):
            network = getNN(len(features[0]), 100)
            start_time = time.time()
            ac[i], f1[i], pr[i], rc[i], ex[i][0], ex[i][1], ex[i][2], ex[i][3] = network.trainNN(features, np.array(classes), x, .5)
            tm[i] = time.time() - start_time
            
        fig+=1
        
        plt.figure(fig)
        plt.plot(num_epochs, ac)
        plt.ylabel('accuracy')
        plt.xlabel('number of epochs (times 10)')
        plt.axis([ 0, num_epochs[ll-1]+500, min(ac), max(ac) ])
        
        fig+=1        
        
        plt.figure(fig)
        plt.plot(num_epochs, f1)
        plt.ylabel('f1')
        plt.xlabel('number of epochs (times 10)')
        plt.axis([ 0, num_epochs[ll-1]+500, min(f1), max(f1) ])
        
        fig+=1        
        
        plt.figure(fig)
        plt.plot(num_epochs, pr)
        plt.ylabel('precision')
        plt.xlabel('number of epochs (times 10)')
        plt.axis([ 0, num_epochs[ll-1]+500, min(pr), max(pr) ])
        
        fig+=1        
        
        plt.figure(fig)
        plt.plot(num_epochs, rc)
        plt.ylabel('recall')
        plt.xlabel('number of epochs (times 10)')
        plt.axis([ 0, num_epochs[ll-1]+500, min(rc), max(rc) ])
        
        fig+=1        
        
        plt.figure(fig)
        lineObjects = plt.plot(num_epochs, ex)
        plt.legend(lineObjects, ('tn', 'tp', 'fn', 'fp'))
        plt.ylabel('examples')
        plt.xlabel('number of epochs (times 10)')
        #plt.axis([ 0, num_epochs[ll-1]+500, 0, 300 ])
        
        fig+=1        
        
        plt.figure(fig)
        plt.plot(num_epochs, tm)
        plt.ylabel('time in seconds')
        plt.xlabel('number of epochs')
        plt.axis([ 0, num_epochs[ll-1]+500, min(tm), max(tm) ])
            
    if bias:
        bias_cut = [.2, .3, .4, .5]
        ll = len(bias_cut)
        ac = [None]*ll
        f1 = [None]*ll
        pr = [None]*ll
        rc = [None]*ll
        ex = np.empty([ll,4])
        tm = [None]*ll
                
        for i, x in enumerate(bias_cut):
            network = getNN(len(features[0]), 100)
            start_time = time.time()
            ac[i], f1[i], pr[i], rc[i], ex[i][0], ex[i][1], ex[i][2], ex[i][3] = network.trainNN(features, np.array(classes), 100, x)
            tm[i] = time.time() - start_time

        fig+=1

        plt.figure(fig)
        plt.plot(bias_cut, ac)
        plt.ylabel('accuracy')
        plt.xlabel('bias cut')
        plt.axis([ 0, bias_cut[0]+.5, min(ac), max(ac) ])
        
        fig+=1        
        
        plt.figure(fig)
        plt.plot(bias_cut, f1)
        plt.ylabel('f1')
        plt.xlabel('bias cut')
        plt.axis([ 0, bias_cut[0]+.5, min(f1), max(f1) ])
        
        fig+=1        
        
        plt.figure(fig)
        plt.plot(bias_cut, pr)
        plt.ylabel('precision')
        plt.xlabel('bias cut')
        plt.axis([ 0, bias_cut[0]+.5, min(pr), max(pr) ])

        fig+=1
        
        plt.figure(fig)
        plt.plot(bias_cut, rc)
        plt.ylabel('recall')
        plt.xlabel('bias cut')
        plt.axis([ 0, bias_cut[0]+.5, min(rc), max(rc) ])
        
        fig+=1        
        
        plt.figure(fig)
        lineObjects = plt.plot(bias_cut, ex)
        plt.legend(lineObjects, ('tn', 'tp', 'fn', 'fp'))
        plt.ylabel('examples')
        plt.xlabel('bias cut')
        #plt.axis([ 0, bias_cut[0]+.5, 0, 300 ])
        
    plt.show()
        
    

args = sys.argv
len_args = len(args)
training = False
extracting = False
metatesting = False

if '-t' in args:
    training = True
    if len_args == 3:
        features_file = args[2]
    if not os.path.isfile(features_file):
        print("Error: Features file doesn't exist.")
        exit();
elif '-e' in args:
    extracting = True
    data = []
    if len_args == 3:
        data_file = args[2]
    elif len_args == 5:
        data_file = args[4]
    with open(data_file, 'r') as df:
        reader = csv.reader(df)
        data = list(reader)
    p = 1
    if args[1] == '-c':
        if args[2].isdigit():
            p = int(args[2])
        else:
            print("The -c parameter should be followed by the number of cores to be used")
            exit()
elif '-m' in args:
    features_file = args[len_args-1]
    if not os.path.isfile(features_file):
        print("Error: Features file doesn't exist.")
        exit();
    metatesting = True
    epochs = False
    nodes = False
    bias = False
    if 'epochs' in args:
        epochs = True
    if 'nodes' in args:
        nodes = True
    if 'bias' in args:
        bias = True
        

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
             
if training or extracting:
    metadata,classes = MetaHandler.get_classified_metadata("metadata.csv","classification.csv")
else:
    metadata = get_metapointer('metadata.csv')
#prune filenames if the system crashed
#prune(filenames, save_file)
#save_file.close()
#open for writing new data
#save_file = open('classes.csv','a')

#EXTRACT FEATURES HERE
if extracting:
    
    #initialize module
    modules = list()
    
    #ADD MODULES HERE
    modules.append(TextScore(True))
    modules.append(BoW_Text_Module(True))
    modules.append(Page_Size_Module())
    modules.append(ScannerDetect())
    modules.append(page_orientation_module())
    modules.append(Bow_Metadata("title"))
    modules.append(Bow_Metadata("author"))
    modules.append(Bow_Metadata("filename"))
    modules.append(Bow_Metadata("folder_name"))
    modules.append(Bow_Metadata("folder_description"))
    modules.append(Bow_Metadata("description"))
    modules.append(Negative_BoW_Text_Module(True))
    modules.append(Resolution_Module())
    
    #NOT WORKING
    #modules.append(OCR_BoW_Module())
    
    print("Extracting Features from the training set. This might take a while...")
    if not data == []:
        extract_features(data, metadata)
    else:
        print("No file data provided.")

#START TRAINING HERE
if training or metatesting:
    
    features, classes, files = load_data(features_file)

    features = [[float(j) for j in i] for i in features]
    
    classes = [float(i) for i in classes]
    
    len_feat = len(features[0])
    
    for i in range(0, len_feat):
        max_nor=max(map(lambda x: x[i], features))
        if max_nor > 1:
            min_nor=min(map(lambda x: x[i], features))
            for f in features: (f[i] - min_nor)/(max_nor-min_nor)
    
    features=np.array([np.array(xi) for xi in features])
    
    from simple_neural_network import NN
    
if training:
    
    print("Initiating Neural Network")
    network = getNN(len(features[0]), 500)
    print("Initialization finished")
    
    print("Starting training.")
    network.trainNN(features, np.array(classes), 500, .5)
    print("Training done!")

#Metatests here
if metatesting:
        
    plot(nodes, epochs, bias, features, classes)

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