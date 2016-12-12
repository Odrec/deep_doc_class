#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 12:04:01 2016

@author: odrec
"""

import sys, os, csv, json
import numpy as np
import pandas as pd
import keras
from subprocess import call
from os.path import basename, dirname, join, splitext, isfile, isdir, exists
from glob import glob  
from multiprocessing import Pool
from doc_globals import*

from Feature_Extractor import FE

def extract_features(files, metadata, c):
    return fe.extract_features(files, metadata, c)
    
def extract_text(files):
    for f in [files]:
        f_id = splitext(basename(f))[0]
        output_txt = join(TXT_PATH,f_id+'.txt')
        if not isfile(output_txt):
            #output_tif = join(TIF_PATH,f_id+'.tif')
            #call(["gs", "-dNOPAUSE", "-sDEVICE=tiffg4", "-r600x600", "-dBATCH", "-sPAPERSIZE=a4", "-sOutputFile="+output_tif, f])
            #call(["tesseract", output_tif, output_txt, '-l', 'eng'])
            call(["gs", "-dNOPAUSE", "-sDEVICE=txtwrite", "-dBATCH", "-q", "-sOutputFile="+output_txt, f])

#def replace_nan_mean(features):
#    features = np.array(features)
#    features = np.where(np.isnan(features), np.ma.array(features, mask=np.isnan(features)).mean(axis=0), features)
#    return features

def replace_nan_mean(features):
    for j, x in enumerate(features):
       if num_files == 1 or batch == 1:
           features[j] = np.float64(x)
       else:
           for i, a in enumerate(x):
               x[i] = np.float64(a)
               
    means=[20298.4197417,2.36381429433,1505.73264143,14.7116948093,0.0620965262834,0.224715647095,0.16046726099,0.0187519213034, \
           0.741276328199,0.302775515611,0.431682559213,0.304774059317,0.320976482497,0.311282582781,0.304324417361,4.1199351857, \
           628.712679189,0.00171229782235,0.0551559240122,0.0781126739863,0.00174256868556,0.148980183858,0.133665986308,0.352290193667]
    features = np.array(features)
    features = np.where(np.isnan(features), means, features)
#    features = np.array(features)
#    col_mean = np.nanmean(features,axis=0)
#    inds = np.where(np.isnan(features))
#    print(inds)
#    print(len(features))
#    print(features[9])
#    print(col_mean)
#    print(inds[0])
#    features[inds]=np.take(col_mean,inds[1])
    return features
    
def norm_features(features):

    max_nor=[6739302.0,1097.62172251,43406.2900391,2064.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.99,0.873788237745,3492.42924243, \
             3840.0,0.0403225806452,0.13734939759,0.258467023173,0.0205552589429,0.404644359115,0.287703016241,1.0]
    min_nor=[0.0,0.0,4.7822265625,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.00313876651982,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

    for j, x in enumerate(features):
       if num_files == 1 or batch == 1:
           f_range = (max_nor[j]-min_nor[j])
           if f_range > 0:
               features[j] = (x-min_nor[j])/f_range
       else:
           for i, a in enumerate(x):
               f_range = (max_nor[i]-min_nor[i])
               if f_range > 0:
                   x[i] = (x[i]-min_nor[i])/f_range
    return features
    
def preprocess_features(features):
    if num_files == 1 or batch == 1:
        lf = len(features)
        doc_ids = features[lf-1]
        features = features[:lf-1]
    else:   
        lf = len(features[0])
        doc_ids = [x[lf-1] for x in features]
        features = [x[:lf-1] for x in features]

    for j, x in enumerate(features):
        if num_files == 1 or batch == 1:
            features[j] = np.float64(x)
        else:
            for i, a in enumerate(x):
                x[i] = np.float64(a)
                
    features = replace_nan_mean(features)
    features = norm_features(features)
    return features, doc_ids
    
#@params test_data a list of numpy arrays. Each array is an input
#@params test_labels a numpy array with the target data
def predict(features, model):
    prd = model.predict(features, verbose=0)
    return prd
        
if __name__ == "__main__":
    args = sys.argv
    len_args = len(args)

    usage = "Usage: classify_pdf.py -m <metadatafile.csv> -d <pdf_path>|<pdf_file> [-c <number_of_cores> -b <number_of_files_per_batch>]"      
    
    if '-h' in args:
        print(usage)
        sys.exit(1)
        
    if not exists(TXT_PATH):
        os.makedirs(TXT_PATH)
        
    if not exists(PREDICTION_PATH):
        os.makedirs(PREDICTION_PATH)
        
    if not len_args == 3 and not len_args == 5 and not len_args == 7 and not len_args == 9:
        print(usage)
        sys.exit(1)
    else:
        metafile = args[2]
        if not '-m' in args or not isfile(metafile):
            m = 2
            print("Warning: No valid metadata file specified. Some features won't be extracted.")
        else:
            m = 0
            metadata = pd.read_csv(join(DATA_PATH,metafile), header=0, delimiter=',', quoting=0, encoding='utf-8')

        if '-d' in args:
            f = args[4-m]
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
    
        num_files = len(files)
        cpus = os.cpu_count()
        
        b = num_files
        c = 1
        if '-b' in args:
            if len_args == 5:
                b = int(args[4])
            elif m == 2 or not '-c' in args:
                b = int(args[6])
            else:
                b = int(args[8])
            if b > num_files:
                b = num_files
        if '-c' in args:
            c = int(args[6-m])
            if c > b:
                c = b
            elif c > cpus:
                c = None

        print("Using %d core(s)"%c)
        print("Using batches of %d files"%b)
        print("Total number of files to process: %d"%num_files)

    path = dirname(files[0])
    #batch = int(math.ceil(len(files)/b))
    batch = b
    over_batch = batch
    under_batch = 0
    num_batch = 0
    fe = FE()
    model=keras.models.load_model("NN.model")
    while(True):
        print("\nBatch %d"%num_batch)
        if num_files < over_batch:
            over_batch = num_files
            
        batch_files = files[under_batch:over_batch]
        batch_meta = {}
        doc_id = []
        for i, f in enumerate(batch_files):
            doc_id.append(splitext(basename(f))[0])
            if m == 0:
                batch_meta[doc_id[i]] = metadata.loc[metadata['document_id'] == doc_id[i]].reset_index(drop=True)

        print("Checking if text needs to be extracted...")
        if num_files == 1 or batch == 1:
            extract_text(batch_files[0])
        else:
            with Pool(c) as pool:
                pool.map(extract_text, batch_files) 
    
        print("Extracting features...")
        features = extract_features(batch_files, batch_meta, c)  
        print("Finished extracting features.")
                    
        print("Preprocessing extracted features...")
        features, doc_id = preprocess_features(features)        
        print("Finished preprocessing features.")
                
        if num_files == 1 or batch == 1:
            features = features[np.newaxis]
        
        print("Predicting classification...")
        predictions = predict(features, model)
        print("Finished prediction.")
        
        prediction_matrix = []
        if num_files == 1 or batch == 1:
            for p in predictions:
                prediction_matrix.append([float(p), doc_id]) 
        else:
            for i, p in enumerate(predictions):
                prediction_matrix.append([float(p), doc_id[i]]) 
        
        output_filename = 'prediction_batch%d'%(num_batch,)
        headers = ["value", "id"]
        with open(join(PREDICTION_PATH, output_filename+".csv"), "w") as f:
            writer = csv.DictWriter(f, fieldnames=headers, delimiter=',')
            writer.writeheader()
            for row in prediction_matrix:
                rowdict = {}
                for i, fname in enumerate(headers):
                    rowdict[fname] = row[i]
                writer.writerow(rowdict)
            
        pred_dict = dict(zip(headers, zip(*prediction_matrix)))
        with open(join(PREDICTION_PATH, output_filename+".json"), 'w') as f:
            json.dump(pred_dict, f)
                
        if over_batch == num_files:
            break
        else:
            under_batch = over_batch
            over_batch = over_batch + batch
            num_batch += 1





