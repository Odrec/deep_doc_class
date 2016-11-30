#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 12:04:01 2016

@author: odrec
"""

import sys, os, csv, json
import numpy as np
from subprocess import call
from os.path import basename, join, splitext, isfile, isdir
from glob import glob  
from multiprocessing import Pool
from doc_globals import*

import Feature_Extractor as fe

def extract_features(files, c):
    return fe.extract_features(files, 'out.csv', c)
    
def extract_text(files):
    for f in [files]:
        f_id = splitext(basename(f))[0]
        output_txt = join(TXT_PATH,f_id+'.txt')
        if not isfile(output_txt):
            #output_tif = join(TIF_PATH,f_id+'.tif')
            #call(["gs", "-dNOPAUSE", "-sDEVICE=tiffg4", "-r600x600", "-dBATCH", "-sPAPERSIZE=a4", "-sOutputFile="+output_tif, f])
            #call(["tesseract", output_tif, output_txt, '-l', 'eng'])
            call(["gs", "-dNOPAUSE", "-sDEVICE=txtwrite", "-dBATCH", "-q", "-sOutputFile="+output_txt, f])
            
#generates the error features
#
#@result:   list of features with the error features included
def generate_error_features(features):
    error_feature = np.zeros((len(features),1))
    for i, x in enumerate(features):
        if np.isnan(x).any():
            error_feature[i] = 1.0              
    [f.extend(error_feature[i]) for i, f in enumerate(features)]
    return features

def replace_nan_mean(features):
    features = np.array(features)
    features = np.where(np.isnan(features), np.ma.array(features, mask=np.isnan(features)).mean(axis=0), features)
    return features
    
def norm_features(features):
    len_feat = len([features[0]])
    max_nor=np.amax(features, axis=0)
    min_nor=np.amin(features, axis=0)
    for i in range(0, len_feat):
        f_range = (max_nor[i]-min_nor[i])
        if(f_range>0):
            features[:,i] = (features[:,i]-min_nor[i])/f_range
        else:
            print(i)
    return features
    
def preprocess_features(features):
    lf = len(features[0])
    ids = [x[lf-1] for x in features]
    features = [x[:lf-2] for x in features]
    for x in features:
        for i, a in enumerate(x):
            x[i] = np.float64(a)
    features = generate_error_features(features)
    features = replace_nan_mean(features)
    return ids, features
    
def predict(features):
    from simple_neural_network import NN
    network = NN()
    return network.predictNN(features)
        
if __name__ == "__main__":
    args = sys.argv
    len_args = len(args)

    usage = "Usage: classify_pdf.py -m <metadatafile.csv> -d <pdf_path>|<pdf_file> [-c <number_of_cores>]"

    c = 1
    if not len_args == 5: 
        if not len_args == 7:
            print(usage)
            sys.exit(1)
        else:
            c = int(args[6])
        
    metafile = args[2]
    if not '-m' in args or not isfile(metafile):
        print("Error: You need to specify a metadata file.")
        print(usage)
        sys.exit(1);        

    f = args[4]
    if '-d' in args:
        if not isfile(f):
            if not isdir(f):
                print("Error: You need to specify a pdf file or path.")
                print(usage)
                sys.exit(1);
            else:
                files = glob(join(f,"*.{}".format('pdf')))
        else:
            files = [f]
    else:
        print("Error: You need to specify a pdf file or path.")
        print(usage)
        sys.exit(1);

    cpus = os.cpu_count()
    num_files = len(files)

    print("Checking if text needs to be extracted...")
    if num_files == 1:
        extract_text(files[0])
    else:
        if c >= cpus and num_files >= cpus:
            pool = Pool()
        elif c > num_files:
            pool = Pool(num_files)
        else:
            pool = Pool(c)
        
        pool.map(extract_text, files) 

    print("Extracting features...")
    features = extract_features(files, c)  
    print("Finished extracting features.")

    print("Preprocessing extracted features...")
    ids, features = preprocess_features(features)
    print("Finished preprocessing features.")
    
    print("Predicting classification...")
    predictions = predict(features)
    print("Finished prediction.")
    
    prediction_matrix = []

    for i, p in enumerate(predictions):
        prediction_matrix.append([float(p), basename(ids[i])]) 
    
    headers = ["value", "id"]
    
    with open(join(RESULT_PATH, "predictions/prediction.csv"),"w") as f:
        writer = csv.DictWriter(f, fieldnames=headers, delimiter=',')
        writer.writeheader()
        for row in prediction_matrix:
            rowdict = {}
            for i, fname in enumerate(headers):
                rowdict[fname] = row[i]
            writer.writerow(rowdict)
        
    pred_dict = dict(zip(headers, zip(*prediction_matrix)))
            
    with open(join(RESULT_PATH, "predictions/prediction.json"), 'w') as f:
        json.dump(pred_dict, f)
            
    print(prediction_matrix)





