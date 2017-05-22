#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 12:04:01 2016

@author: odrec

Main script for classifying copyrighted pdf documents.

Usage:
    classify_pdf.py [-fp <pdf_path>|<pdf_file> -conf <config_file> 
    -meta <metadatafile.csv> || <filename=<filename>,folder_name=<folder_name>> -mod <trained_model>
    -c <number_of_cores> -b <number_of_files_per_batch> -sp -sf -pf <preprocessing_file> -ff <features_file> 
    -rf <predictions_file> -preprocess_only -features_only -t <threshold_for_classification>]
    
Parameters:
    -fp: parameter used to specify the path to the pdf file(s). This parameter is always required
    
    -conf: parameter used to pass the config file. If a config file is passed then the values specified in it
    will take precedence over the parameters given in the command line. Each parameter must be specified on a new line
    with the name of the parameter, if the parameter has a value, the name should be followed by an equal sign (=)
    and then the value of the parameter. Ex. metadata_file=../metadata.csv or save_preprocess.
    Parameters that can be specified on the config file:
        metadata_file: path to metadata csv file
        batch: the quantity of files per batch
        model: path to trained model
        cores: number of cores to be used for parallel processing
        predict_threshold: the threshold used for classification of the documents
        save_preprocess: use this parameter if the preprocessing data should be saved on your hard disc
        save_features: use this parameter if the features should be saved on your hard disc
        preprocess_only: use this parameter if only the preprocessing data should be extracted and saved on your hard disc
        features_only: use this parameter if only the feature data should be calculated and saved on your hard disc
        preprocessing_file: specifies an existing file on which to append the preprocessing data.
        features_file: specifies an existing file on which to append the feature data.
        prediction_file: specifies an existing file on which to append the result predicition data.
    
    -meta: parameter used to specify the path to the metadata csv file. It is also possible to pass the metadata of 
    a single file directly on the command line by writing filename=<filename>,folder_name=<folder_name> instead of
    the path to the metadata csv file. Be aware that if the metadata is passed on the command line the -fp parameter should point
    to one single file and not to a path of a group of files.
    
    -mod: parameter used to specify the path to the trained model. If no model is specified the default ones will be loaded.
    The default model with metadata features is NN.model, the default model without metadata features is NN_noMeta.model.
    
    -c: parameter used to specify the number of cores to be used for parallel processing. 
    
    -b: parameter used to specify the number of files to be processed per batch. The preprocessing, features and prediction
    results will be updated after each batch on the saving files.
    
    -sp: parameter used if you want to save the preprocessing data. If it doesn't exist a folder will be created in
    '../preprocessing data'. Inside this path a 'text_files' folder will be created to store the extracted text 
    from each file and a 'features' folder will be created to store the features.
    
    -sf: parameter used if you want to save the features data.
    
    -pf: parameter used to specify the preprocessing file. If the file doesn't exist it will be created.
    The default file is 'preprocessing_data/preprocessing_data.json'. If you don't use this argument the 
    existing default file will be ovewritten.
    
    -ff: parameter used to specify the features file. If the file doesn't exist it will be created.
    The default file is 'preprocessing_data/features/features.json'. If you don't use this argument
    the existing default file will be ovewritten.
    
    -rf: parameter used to specify the result predictions file. If the file doesn't exist it will be created. The default
    file if this parameter is not specified is '../predictions/prediction.json'. If you don't use this argument
    the existing default file will be ovewritten.
    
    -preprocess_only: parameter used if you want to extract and save preprocessing data only.
    
    -features_only: parameter used if you want to extract and save features data only.
    
    -t: parameter used to specify the threshold for classification. The deafult value is 0.5.
    
"""

import sys, json
import numpy as np
from os.path import join
import param as pa
import data as da
import model as mo
from pprint import pprint

FEATURES_NAMES = ['file_size', 'creator', 'avg_ib_size', 'avg_tb_size', 'error', 'pages', 
'ratio_words_tb', 'folder_name', 'ratio_tb_ib', 'ratio_tb_pages', 'producer', 'text', 
'ratio_ib_pages', 'filename', 'page_rot']

FEATURES_NAMES_NO_METADATA = ['file_size', 'creator', 'avg_ib_size', 'avg_tb_size', 'error', 'pages', 
'ratio_words_tb', 'ratio_tb_ib', 'ratio_tb_pages', 'producer', 'text', 
'ratio_ib_pages', 'page_rot']
          
if __name__ == "__main__":
    args = sys.argv
    len_args = len(args)

    usage = "Usage: classify_pdf.py [-fp [PATH]|[FILE]] [-conf [FILE]] \
    [-meta [FILE] or [filename=<filename>,folder_name=<folder_name>]] [-mod [FILE]] \
    [-c [INT]] [-b [INT]] [-sp] [-sf] [-pf [FILE]] [-ff [FILE]]\
    [-rf [FILE]] [-preprocess_only] [-features_only] [-t [FLOAT]]\n\n\
    The -fp option to specify the path to pdf files is the only required argument.\n\n\
    optional arguments:\n\
    -h, --help: prints this help\n\
    -conf: config file\n\
    -meta: metadata csv file\n\
    -mod: trained model\n\
    -c: number of cores\n\
    -b: batch quantity\n\
    -sp: save the preprocessing data\n\
    -sf: save the features data\n\
    -pf: preprocessing file\n\
    -ff: features file\n\
    -rf: result predictions file\n\
    -preprocess_only: extract and save preprocessing data only\n\
    -features_only: extract and save features data only\n\
    -t: threshold for classification"      
    
    if '-h' in args or '--help' in args:
        print(usage)
        sys.exit(1)
        
    print("Setting up parameters and environment...")
    
    config_file, config_data = pa.load_config_file(args)
    if config_data:
        config_metadata_file = config_data[0]
        config_batch = config_data[1]
        config_model = config_data[2]
        config_cores = config_data[3]
        config_threshold = config_data[4]
        config_savepreproc = config_data[5]
        config_savefeatures = config_data[6]
        config_preproconly = config_data[7]
        config_featuresonly = config_data[8]
        config_preprocessing_file = config_data[9]
        config_features_file = config_data[10]
        config_prediction_file = config_data[11]
    else:
        print("Error: You need to specify a valid config file.")
        print(config_file,": is not valid configuration file!")
        print(usage)
        sys.exit(1)
        
    metadata, metafile, doc_ids, pdf_path = pa.get_metafile(args, config_metadata_file)
    if metadata is None:
        print("Error: You need to specify a valid csv file with the metadata.")
        print(metafile,": is not a valid csv metadata file!")
        print(usage)
        sys.exit(1)
    if pdf_path == -1:
        print("Error: You need to specify a valid path for the pdf file(s).")
        print("No valid files path specified for this metadata:",metafile,". Use the -fp parameter.")
        print(usage)
        sys.exit(1)
        
    pdf_path, doc_ids = pa.get_pdf_path_files(args, doc_ids, pdf_path)
    if pdf_path is None:
        print("Error: you need to specify a valid pdf file or a valid path to a group of pdf files.")
        print(pdf_path,": is an invalid path to pdf files or an invalid pdf file!")
        print(usage)
        sys.exit(1) 
           
    num_files = len(doc_ids)

    check_batch, batch_quantity = pa.get_batch_value(args, num_files, config_batch)
    if check_batch == -1:
        print("Error: You need to specify a valid batch value.")
        print(batch_quantity,": is not a valid batch value!")
        print(usage)
        sys.exit(1)
           
    check_cores, cores = pa.get_number_of_cores(args, batch_quantity, config_cores)
    if check_cores == -1:
        print("Error: You need to specify a valid quantity of cores.")
        print(cores,": is not a valid quantity of cores!")
        print(usage)
        sys.exit(1)
    
    check_threshold, decision_threshold = pa.get_decision_threshold(args, config_threshold)
    if check_threshold == -1:
        print("Error: You need to specify a valid threshold between 0.0 and 1.0.")
        print(decision_threshold,": is not a valid value!")
        print(usage)
        sys.exit(1)
        
    check_paths, preprocessing_file, features_file, prediction_file = pa.create_result_folders_files(args, config_preprocessing_file,
                                                                                config_features_file, config_prediction_file)
    if check_paths == -1:
        print("Error: Failed to create path.")
        print(preprocessing_file,": this path or subpath couldn't be created!")
        print(usage)
        sys.exit(1)
    if check_paths == -2:
        print("Error: Failed to create file.")
        print(prediction_file,": this file couldn't be created!")
        print(usage)
        sys.exit(1)
    
    save_preproc, preproc_file, save_feature, \
    feature_file, preprocess_only, features_only = pa.set_save_preprocess_features(args, config_savepreproc, 
                                                                                   config_savefeatures, 
                                                                                   config_preproconly, 
                                                                                   config_featuresonly)
    if save_preproc == -1:
        print("Error: You need to specify an existing preprocessing json file.")
        print(preproc_file,": this file doesn't exist!")
        print(usage)
        sys.exit(1)
    if save_feature == -1:
        print("Error: You need to specify an existing features json file.")
        print(feature_file,": this file doesn't exist!")
        print(usage)
        sys.exit(1)

    print("Using %d core(s)"%cores)
    print("Using batches of %d files"%batch_quantity)
    print("Total number of files to process: %d"%num_files)
    
    print("Loading trained module...")
    model, model_path = mo.load_trained_model(args, config_model, metadata )
    if model is None:
        print("Some error ocurred during the loading of the trained module.")
        print("Please check if the module file exists and if it's valid.")
        print("Failed to load model: ",model_path)
        print(usage)
        sys.exit(1) 
        
    print("Initiating the extraction of features and classification process...")
    files = []
    for d in doc_ids:
        files.append(join(pdf_path,d+'.pdf'))
    over_batch = batch_quantity
    under_batch = 0
    num_batch = 0
                    
    #try:
    while(True):
        print("\nBatch %d"%num_batch)
        if num_files < over_batch:
            over_batch = num_files
            
        batch_files = files[under_batch:over_batch]
        if batch_quantity <= 10:
            print("Files to process: ")
            pprint(batch_files)
        print("Preprocessing data...")
        print("Checking if data needs to be extracted and extracting it if needed.")
        print("WARNING: this could take a long time.")
        ids, preprocessing_list = da.get_preprocessing_data(batch_files, metadata, cores)
        print("Finished getting preprocessing data.")
            
        if save_preproc:
            print("Saving the extracted pre-processing data...")
            da.save_preproc_data(ids, preprocessing_list, preprocessing_file)
            
        if not preprocess_only:
            print("Extracting features...")
            print("WARNING: this could take a long time.")
            features_dict = da.get_features(ids, preprocessing_list, preprocessing_list[0])
            print("Finished extracting features.")
                        
            print("Preprocessing extracted features...")
            da.normalize_features(features_dict)        
            print("Finished preprocessing features.")
            
            if save_feature:
                print("Saving the features...")
                da.save_features(features_dict, features_file)

            if not features_only:
                features_names = []
                if metadata:
                    features_names = FEATURES_NAMES
                else:
                    features_names = FEATURES_NAMES_NO_METADATA
                input_list = [[None]*len(features_names) for _ in range(len(features_dict.keys()))]
                item = 0
                for key, feats in features_dict.items():
                    for f, value in feats.items():
                        index_f = features_names.index(f)
                        input_list[item][index_f] = value
                    item += 1
                input_list = np.array(input_list)
                print("Predicting classification...")
                predictions = mo.predict(input_list, model)
                print("Finished prediction.")
                
                prediction_matrix_batch = {}
                for i, p in enumerate(predictions):
                    pred = float(predictions[i])
                    prediction_matrix_batch[ids[i]] = {}
                    prediction_matrix_batch[ids[i]]['value'] = pred
                    if pred > decision_threshold:
                        prediction_matrix_batch[ids[i]]['class'] = 1 
                    else:
                        prediction_matrix_batch[ids[i]]['class'] = 0 
                                        
                print("Saving results for batch %d to json file..."%num_batch)
                try:
                    with open(join(prediction_file), "r") as jsonFile:
                        data = json.load(jsonFile)
                    data.update(prediction_matrix_batch)
                except:
                    data = prediction_matrix_batch               
                with open(join(prediction_file), "w") as jsonFile:
                    json.dump(data, jsonFile)
                
        if over_batch == num_files:
            print("Finished processing all files...")
            break
        else:
            under_batch = over_batch
            over_batch = over_batch + batch_quantity
            num_batch += 1
    #except: pass





