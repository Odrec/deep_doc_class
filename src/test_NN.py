#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 14:14:12 2016

@author: odrec
"""
import sys
from os.path import join
import matplotlib.pyplot as plt
import time
import csv
import numpy as np
from doc_globals import*

from simple_neural_network import NN

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

#loads the features from the csv file created during extraction
#
#@result:   list of lists of features, list of corresponding classes and filenames
def load_data(features_file='output_complete.csv'):
    
    with open(features_file, 'r') as f:
      reader = csv.reader(f)
      data = list(reader)
      data = data[1:]
      
    num_features = len(data[0])-2
            
    features = [item[:num_features] for item in data]

    features = generate_error_features(features)     
    
    classes = [item[num_features] for item in data]
    filenames = [item[num_features+1] for item in data]

    features = [[float(j) for j in i] for i in features]
    
    classes = [float(i) for i in classes]
    
    len_feat = len(features[0])
    
    for i in range(0, len_feat):
        max_nor=max(map(lambda x: x[i], features))
        if max_nor > 1:
            min_nor=min(map(lambda x: x[i], features))
            for f in features: (f[i] - min_nor)/(max_nor-min_nor)
    
    features=np.array([np.array(xi) for xi in features])
    
    return features, classes, filenames
    

#this function will initialize the neural network
# @parram input_dim:    number of modules for the input vector
def getNN(input_dim, hidden_dim, num_hidden_layers=1):
    network = NN()
    network.initializeNN(input_dim, hidden_dim, num_hidden_layers)
    return network
    
def test_NN(file=join(FEATURE_VALS_PATH,'output_complete.csv')):
    
        num_epochs = [1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
        ll = len(num_epochs)
        ac = [None]*ll
        f1 = [None]*ll
        pr = [None]*ll
        rc = [None]*ll
        ex = np.empty([ll,4])
        tm = [None]*ll
        ac2 = [None]*ll
        f12 = [None]*ll
        pr2 = [None]*ll
        rc2 = [None]*ll
        ex2 = np.empty([ll,4])
        tm2 = [None]*ll

        fig = 0

        features, classes, filenames = load_data(file)
                
        for i, x in enumerate(num_epochs):
            network = getNN(len(features[0]), 100, 1)
            network2 = getNN(len(features[0]), 100, 2)
            start_time = time.time()
            ac[i], f1[i], pr[i], rc[i], ex[i][0], ex[i][1], ex[i][2], ex[i][3] = network.trainNN(features, np.array(classes), x, .5, 2)
            tm[i] = time.time() - start_time
            start_time = time.time()
            ac2[i], f12[i], pr2[i], rc2[i], ex2[i][0], ex2[i][1], ex2[i][2], ex2[i][3] = network2.trainNN(features, np.array(classes), x, .5, 2)
            tm2[i] = time.time() - start_time
            
        fig+=1
        
        plt.figure(fig)
        plt.plot(num_epochs, ac)
        plt.ylabel('accuracy (%)')
        plt.xlabel('number of epochs (times 10)')
        plt.axis([ 0, num_epochs[ll-1]+(num_epochs[ll-1]*.1), min(ac)-.5, max(ac)+.5 ])
        
        plt.savefig(join(RESULT_PATH, 'epochsAc.jpg'))
        
        fig+=1        
        
        plt.figure(fig)
        plt.plot(num_epochs, f1)
        plt.ylabel('f1')
        plt.xlabel('number of epochs (times 10)')
        plt.axis([ 0, num_epochs[ll-1]+(num_epochs[ll-1]*.1), min(f1)-.1, max(f1)+.1 ])
        
        plt.savefig(join(RESULT_PATH, 'epochsF1.jpg'))
        
        fig+=1        
        
        plt.figure(fig)
        plt.plot(num_epochs, pr)
        plt.ylabel('precision')
        plt.xlabel('number of epochs (times 10)')
        plt.axis([ 0, num_epochs[ll-1]+(num_epochs[ll-1]*.1), min(pr)-.1, max(pr)+.1 ])
        
        plt.savefig(join(RESULT_PATH, 'epochsPr.jpg'))
        
        fig+=1        
        
        plt.figure(fig)
        plt.plot(num_epochs, rc)
        plt.ylabel('recall')
        plt.xlabel('number of epochs (times 10)')
        plt.axis([ 0, num_epochs[ll-1]+(num_epochs[ll-1]*.1), min(rc)-.1, max(rc)+.1 ])
        
        plt.savefig(join(RESULT_PATH, 'epochsRc.jpg'))
        
        fig+=1        
        
        plt.figure(fig)
        lineObjects = plt.plot(num_epochs, ex)
        plt.legend(lineObjects, ('tn', 'tp', 'fn', 'fp'))
        plt.ylabel('number of examples')
        plt.xlabel('number of epochs (times 10)')
        #plt.axis([ 0, num_epochs[ll-1]+500, 0, 300 ])
        
        plt.savefig(join(RESULT_PATH, 'epochsEx.jpg'))
        
        fig+=1        
        
        plt.figure(fig)
        plt.plot(num_epochs, tm)
        plt.ylabel('time in seconds')
        plt.xlabel('number of epochs')
        plt.axis([ 0, num_epochs[ll-1]+(num_epochs[ll-1]*.1), min(tm), max(tm) ])
        
        plt.savefig(join(RESULT_PATH, 'epochsTm.jpg'))
        
        fig+=1
        
        plt.figure(fig)
        plt.plot(num_epochs, ac2)
        plt.ylabel('accuracy (%)')
        plt.xlabel('number of epochs (times 10)')
        plt.axis([ 0, num_epochs[ll-1]+(num_epochs[ll-1]*.1), min(ac2)-.5, max(ac2)+.5 ])
        
        plt.savefig(join(RESULT_PATH, 'epochsAc2.jpg'))
        
        fig+=1        
        
        plt.figure(fig)
        plt.plot(num_epochs, f12)
        plt.ylabel('f1')
        plt.xlabel('number of epochs (times 10)')
        plt.axis([ 0, num_epochs[ll-1]+(num_epochs[ll-1]*.1), min(f12)-.1, max(f12)+.1 ])
        
        plt.savefig(join(RESULT_PATH, 'epochsF12.jpg'))
        
        fig+=1        
        
        plt.figure(fig)
        plt.plot(num_epochs, pr2)
        plt.ylabel('precision')
        plt.xlabel('number of epochs (times 10)')
        plt.axis([ 0, num_epochs[ll-1]+(num_epochs[ll-1]*.1), min(pr2)-.1, max(pr2)+.1 ])
        
        plt.savefig(join(RESULT_PATH, 'epochsPr2.jpg'))
        
        fig+=1        
        
        plt.figure(fig)
        plt.plot(num_epochs, rc2)
        plt.ylabel('recall')
        plt.xlabel('number of epochs (times 10)')
        plt.axis([ 0, num_epochs[ll-1]+(num_epochs[ll-1]*.1), min(rc2)-.1, max(rc2)+.1 ])
        
        plt.savefig(join(RESULT_PATH, 'epochsRc2.jpg'))
        
        fig+=1        
        
        plt.figure(fig)
        lineObjects = plt.plot(num_epochs, ex2)
        plt.legend(lineObjects, ('tn', 'tp', 'fn', 'fp'))
        plt.ylabel('number of examples')
        plt.xlabel('number of epochs (times 10)')
        #plt.axis([ 0, num_epochs[ll-1]+500, 0, 300 ])
        
        plt.savefig(join(RESULT_PATH, 'epochsEx2.jpg'))
        
        fig+=1        
        
        plt.figure(fig)
        plt.plot(num_epochs, tm2)
        plt.ylabel('time in seconds')
        plt.xlabel('number of epochs')
        plt.axis([ 0, num_epochs[ll-1]+(num_epochs[ll-1]*.1), min(tm2), max(tm2) ])
        
        plt.savefig(join(RESULT_PATH, 'epochsTm2.jpg'))
        
args = sys.argv
test_NN(args[1])