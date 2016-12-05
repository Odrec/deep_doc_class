#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 12:04:01 2016

@author: odrec
"""

import sys, os, csv, json, math
import numpy as np
from subprocess import call
from os.path import basename, dirname, join, splitext, isfile, isdir
from glob import glob  
from multiprocessing import Pool
from doc_globals import*

from Feature_Extractor import FE
import MetaHandler as MH

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
    if num_files == 1 or batch == 1:
        lf = len(features)
        features = features[:lf-2]
    else:   
        lf = len(features[0])
        features = [x[:lf-2] for x in features]

    for j, x in enumerate(features):
        if num_files == 1 or batch == 1:
            features[j] = np.float64(x)
        else:
            for i, a in enumerate(x):
                x[i] = np.float64(a)
                
    features = replace_nan_mean(features)
    return features
    
def predict(features):
    from simple_neural_network import NN
    network = NN()
    return network.predictNN(features)
        
if __name__ == "__main__":
    args = sys.argv
    len_args = len(args)

    usage = "Usage: classify_pdf.py -m <metadatafile.csv> -d <pdf_path>|<pdf_file> [-c <number_of_cores> -b <number_of_files_per_batch>]"      
    
    if '-h' in args:
        print(usage)
        sys.exit(1)
        
    if not len_args == 3 and not len_args == 5 and not len_args == 7 and not len_args == 9:
        print(len_args)
        print(usage)
        sys.exit(1)
    else:
        if not '-m' in args or not isfile(metafile):
            m = 2
            print("Warning: No valid metadata file specified. Some features won't be extracted.")
        else:
            m = 0
            metafile = args[2]
            metadata = MH.get_classified_meta_dataframe(metafile)

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
            if len_args == 7:
                b = int(args[6-m])
            else: b = int(args[8-m])
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

    path = dirname(files[0])
    #batch = int(math.ceil(len(files)/b))
    batch = b
    over_batch = batch
    under_batch = 0
    num_batch = 0
    fe = FE()
    while(True):
        print("\nBatch %d"%num_batch)
        if num_files < over_batch:
            over_batch = num_files
            
        batch_files = files[under_batch:over_batch]
        batch_meta = []
        doc_id = []
        for i, f in enumerate(batch_files):
            doc_id.append(splitext(basename(f))[0])
            if m == 0:
                batch_meta.append(metadata.loc[metadata['document_id'] == doc_id[i]].reset_index(drop=True))

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
        features = preprocess_features(features)
        print("Finished preprocessing features.")
        
        if num_files == 1 or batch == 1:
            features = features[np.newaxis]
        
        print("Predicting classification...")
        predictions = predict(features)
        print("Finished prediction.")
        
        prediction_matrix = []
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





