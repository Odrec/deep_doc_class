#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 11:09:01 2017

@author: odrec
"""

import os, csv
from os.path import realpath, basename, dirname, join, splitext, isfile, isdir
from glob import glob  

def create_result_folders_files(args, config_preprocessing_file, config_features_file, config_prediction_file):
    '''
    Creates folders and files necessary for the execution of the program
    
    @param args: list of arguments passed as parameters
    @dtype args: list

    @param config_preprocessing_file: preprocessing file specified in the config file
    @dtype config_preprocessing_file: str
    
    @param config_features_file: features file specified in the config file
    @dtype config_features_file: str
    
    @param config_prediction_file: prediction file specified in the config file
    @dtype config_prediction_file: str
    
    @return check_flag: flag to check if all the folders and files were succesfully created
    @rtype check_flag: int [-2:0]
    
    @return preprocessing_file: the created preprocessing file
    @rtype preprocessing_file: str
    @NOTE: if there's an error this return value will store which path or file the error happened on
    
    @return features_file: the created features file
    @rtype features_file: str
    
    @return prediction_file: the created preprocessing file
    @rtype prediction_file: str
    '''
    check_flag = 0
    preprocessing_path = join(join(dirname(realpath(__file__)), os.pardir), "preprocessing_data")
    text_path = join(preprocessing_path, "text_files")
    features_path = join(preprocessing_path, "features")
    prediction_path = join(join(dirname(realpath(__file__)), os.pardir), "predictions")
    preprocessing_file = False
    features_file = False
    prediction_file = False
    overwrite_preproc = False
    overwrite_features = False
    overwrite_prediction = False
    if not config_preprocessing_file == 'placeholder':
        preprocessing_file = config_preprocessing_file
    if not config_features_file == 'placeholder':
        features_file = config_features_file
    if not config_prediction_file  == 'placeholder':
        prediction_file = config_prediction_file

    if not preprocessing_file:
        if '-pf' in args:
            index_preprocess = args.index('-pf') + 1
            preprocessing_file = args[index_preprocess]
        else:
            preprocessing_file = 'preprocessing_data.json'
            overwrite_preproc = True
    if not features_file:
        if '-ff' in args:
            index_features = args.index('-ff') + 1
            features_file = args[index_features]
        else:
            features_file = 'features.json'
            overwrite_features = True
    if not prediction_file:
        if '-rf' in args:
            index_prediction = args.index('-rf') + 1
            prediction_file = args[index_prediction]
        else:
            prediction_file = 'prediction.json'
            overwrite_prediction = True
    if not isdir(preprocessing_path):
        try:
            os.makedirs(preprocessing_path)
            os.makedirs(text_path)
            os.makedirs(features_path)
        except:
            check_flag = -1
            return check_flag, preprocessing_path, features_file, prediction_file
    if not isdir(text_path):
        try:
            os.makedirs(text_path)
        except:
            check_flag = -1
            return check_flag, text_path, features_file, prediction_file
    if not isdir(features_path):
        try:
            os.makedirs(features_path)
        except:
            check_flag = -1
            return check_flag, features_path, features_file, prediction_file
    if not isdir(prediction_path):
        try:
            os.makedirs(prediction_path)
        except:
            check_flag = -1
            return check_flag, prediction_path, features_file, prediction_file
    if overwrite_preproc: preprocessing_file = join(preprocessing_path, preprocessing_file)
    if not isfile(preprocessing_file) or overwrite_preproc:
        try:
            open(preprocessing_file, 'w').close()
        except:
            check_flag = -2
            return check_flag, preprocessing_file, features_file, prediction_file
    if overwrite_features: features_file = join(features_path, features_file)
    if not isfile(features_file) or overwrite_features:
        try:
            open(features_file, 'w').close()
        except:
            check_flag = -2
            return check_flag, features_file, features_file, prediction_file
    if overwrite_prediction: prediction_file = join(prediction_path, prediction_file)
    if not isfile(prediction_file) or overwrite_prediction:
        try:
            open(prediction_file, 'w').close()
        except:
            check_flag = -2
            return check_flag, prediction_file, features_file, prediction_file     
    
    return check_flag, preprocessing_file, features_file, prediction_file
        
def load_config_file(args): 
    '''
    Loads the parameters from the configuration file
    
    @param args: list of arguments passed as parameters
    @dtype args: list
    
    @return config_file: the configuration file
    @rtype config_file: str
    
    @return config_params: all the parameters loaded from the configuration file
    @rtype config_params: list
    '''
    config_metadata_file, config_batch, config_model, config_cores, config_threshold, config_savepreproc,\
    config_savefeatures, config_preproconly, config_featuresonly, config_preprocessing_file, config_features_file,\
    config_prediction_file, config_file = ('placeholder',)*13
    if '-conf' in args:
        print("Using config file...")
        index_conf = args.index('-conf')+1
        config_file = args[index_conf] 
        if not isfile(config_file):
            config_file = 'param.conf'
        if isfile(config_file):
            configuration_list = [line.strip() for line in open(config_file).readlines()]
            configuration_list = [line.split('=') for line in configuration_list]
            for l in configuration_list:
                if l[0] == 'metadata_file':
                    config_metadata_file = l[1]
                elif l[0] == 'batch':
                    config_batch = l[1]
                elif l[0] == 'model':
                    config_model = l[1]
                elif l[0] == 'cores':
                    config_cores = l[1]
                elif l[0] == 'predict_threshold':
                    config_threshold = l[1]
                elif l[0] == 'save_preprocess':
                    config_savepreproc = True
                elif l[0] == 'save_features':
                    config_savefeatures = True
                elif l[0] == 'preprocess_only':
                    config_preproconly = True
                elif l[0] == 'features_only':
                    config_featuresonly = True
                elif l[0] == 'preprocessing_file':
                    config_preprocessing_file = l[1]
                elif l[0] == 'features_file':
                    config_features_file = l[1]
                elif l[0] == 'prediction_file':
                    config_prediction_file = l[1]
        else:
            return config_file, False
                
    config_params = [config_metadata_file, config_batch, config_model, config_cores, config_threshold, 
            config_savepreproc, config_savefeatures, config_preproconly, config_featuresonly, config_preprocessing_file, 
            config_features_file, config_prediction_file]
    return config_file, config_params
    
def get_metafile(args, config_metadata_file):
    '''
    Gets the metadata file and extracts the metadata
    
    @param args: list of arguments passed as parameters
    @dtype args: list
    
    @param config_metadata_file: metadata file specified in the config file
    @dtype config_metadata_file: str
    
    @return metadata: metadata for all files
    @rtype metadata: dict
    
    @return metafile: path to the metadata file
    @rtype metafile: str
    
    @return doc_ids: ids for all the documents
    @rtype doc_ids: list
    
    @return pdf_path: path to the pdf files
    @rtype pdf_path: str
    '''
    metafile = -1
    if isfile(config_metadata_file):
        metafile = config_metadata_file
    elif '-meta' in args:
        index_meta = args.index('-meta')+1
        metafile = args[index_meta]
    doc_ids = []
    metadata = {}
    pdf_path = None
    if isfile(metafile): 
        if '-fp' in args:
            index_pdf = args.index('-fp')+1
            pdf_path = args[index_pdf]
            if isfile(pdf_path) and splitext(basename(pdf_path))[1] == '.pdf':
                pdf_files = [pdf_path]
            elif isdir(pdf_path):
                pdf_files = glob(join(pdf_path,"*.{}".format('pdf')))
            else:
                return metadata, metafile, doc_ids, -1
            dids_folder = []
            for f in pdf_files: dids_folder.append(splitext(basename(f))[0])
            if splitext(basename(metafile))[1] == '.csv':
                reader = csv.DictReader(open(metafile, 'r'))
                metadata_feature_list = ['filename', 'folder_name']
                for row in reader:
                    key = row.pop('document_id')
                    if key in dids_folder:
                        doc_ids.append(key)
                        entrydict = {}
                        for f in metadata_feature_list:
                            entrydict[f] = row[f]
                        metadata[key] = entrydict
            else: 
                return None, metafile, doc_ids, pdf_path
        else: return metadata, metafile, doc_ids, -1
    elif isinstance(metafile, str): 
        flag_pdf = False
        if '-fp' in args:
            index_pdf = args.index('-fp')+1
            pdf = args[index_pdf]
            if isfile(pdf) and splitext(basename(pdf))[1] == '.pdf':
                flag_pdf = True
                pdf_path = dirname(pdf)
                pdf = splitext(basename(pdf))
                fileid = pdf[0]
        if flag_pdf:
            doc_ids.append(fileid)
            metadata[fileid] = {}
            metadata_list = metafile.replace('=',' ').replace(',',' ').split()
            index_filename = metadata_list.index('filename')+1
            metadata[fileid]['filename'] = metadata_list[index_filename]
            index_foldername = metadata_list.index('folder_name')+1
            metadata[fileid]['folder_name'] = metadata_list[index_foldername]
        else:
            return metadata, metafile, doc_ids, -1
    else:
        print("Warning: No metadata specified. Some of the features won't be extracted.")
        metadata = {}

    return metadata, metafile, doc_ids, pdf_path
    
def get_pdf_path_files(args, doc_ids, pdf_path):
    '''
    Gets all the pdf files
    
    @param args: list of arguments passed as parameters
    @dtype args: list
    
    @param doc_ids: ids for all the documents
    @dtype doc_ids: list
    
    @param pdf_path: path to the pdf files
    @dtype pdf_path: str
    
    @return pdf_path: path to pdf files
    @rtype pdf_path: str
    
    @return doc_ids: ids of the documents
    @rtype doc_ids: list
    '''
    if doc_ids:
        if isdir(pdf_path):
            return pdf_path, doc_ids
    doc_ids = []
    pdf_path = None
    if '-fp' in args:
        index_files = args.index('-fp')+1
        pdf_path = args[index_files]
    else: return pdf_path, doc_ids
    pdf = splitext(basename(pdf_path))
    if isfile(pdf_path) and pdf[1] == '.pdf':
        doc_ids = [pdf[0]]
        pdf_path = dirname(pdf_path)
    elif isdir(pdf_path):
        dids = glob(join(pdf_path,"*.{}".format('pdf')))
        for d in dids: doc_ids.append(splitext(basename(d))[0])        
    else:
        return pdf_path, doc_ids
                
    return pdf_path, doc_ids
    
def get_batch_value(args, num_files, config_batch):
    '''
    Gets the value for batch
    
    @param args: list of arguments passed as parameters
    @dtype args: list
    
    @param num_files: the total number of files
    @dtype num_files: int
    
    @param config_batch: the quantity for batch specified in the config file
    @dtype config_batch: str
    
    @return check_flag: flag to check if all the batch value is valid
    @rtype check_flag: int [-1:0]
    
    @return batch_quant: quantity of batch files
    @rtype batch_quant: int
    '''
    check_flag = 0
    if config_batch.isdigit():
        return check_flag, int(config_batch)
    batch_quant = '1'
    if '-b' in args:
        index_batch = args.index('-b')+1
        batch_quant = args[index_batch]
    if batch_quant.isdigit():
        batch_quant = int(batch_quant)
        if num_files < batch_quant:
            print("Using total number of files as value of batch...")
            batch_quant = num_files
    else:
        check_flag = -1
        return check_flag, batch_quant
            
    return check_flag, batch_quant
    
def get_number_of_cores(args, batch_quant, config_cores):
    '''
    Gets the value for the number of cores to be used
    
    @param args: list of arguments passed as parameters
    @dtype args: list
    
    @param batch_quant: the number of files in a batch
    @dtype batch_quant: int
    
    @param config_cores: the value of cores specified in the config file
    @dtype config_cores: str
    
    @return check_flag: flag to check if all the batch value is valid
    @rtype check_flag: int [-1:0]
    
    @return cores: quantity of cores
    @rtype cores: int
    '''
    check_flag = 0
    if config_cores.isdigit():
        return check_flag, int(config_cores)
    cores = '1'
    if '-c' in args:
        index_cores = args.index('-c') + 1
        cores = args[index_cores]
    cpus = os.cpu_count()
    if cores.isdigit():
        cores = int(cores)
        if cores > batch_quant:
            print("Using the number of files per batch for the number of cores...")
            cores = batch_quant
        elif cores > cpus:
            print("Using the maximum number of cores available...")
            print("WARNING: this could limit the performance of your computer considerably.")
            cores = None
    else:
        check_flag = -1
        return check_flag, cores
        
    return check_flag, cores
    
def get_decision_threshold(args, config_threshold):
    '''
    Gets the value for the decicion threshold
    
    @param args: list of arguments passed as parameters
    @dtype args: list
    
    @param config_threshold: the value for the decicion threshold specified in the config file
    @dtype config_thresold: str
    
    @return check_flag: flag to check if all the batch value is valid
    @rtype check_flag: int [-1:0]
    
    @return decision_threshold: value for the decision threshold
    @rtype decision_threshold: float
    '''
    check_flag = 0
    try:
        check_flag = 1
        return check_flag, float(config_threshold)
    except:
        decision_threshold = 0.5        
        if '-t' in args:
            index_threshold = args.index('-t') + 1
            try:
                decision_threshold = float(args[index_threshold])
                if decision_threshold > 1 or decision_threshold < 0:
                    check_flag = -1
                    return check_flag, decision_threshold
            except:
                check_flag = -1
                return check_flag, decision_threshold
    return check_flag, decision_threshold

def set_save_preprocess_features(args, config_savepreproc, config_savefeatures, config_preproconly, config_featuresonly):
    '''
    @param args: list of arguments passed as parameters
    @dtype args: list
    
    @param config_savepreproc: save preprocessing data is True if it was in the config file
    @dtype config_savepreproc: bool/str
    
    @param config_savefeatures: save features is True if it was in the config file
    @dtype config_savefeatures: bool/str

    @param config_preproconly: only extract preprocessing data is True if it was in the config file
    @dtype config_preproconly: bool/str

    @param config_featuresonly: only extract features is True if it was in the config file
    @dtype config_featuresonly: bool/str
    
    @return save_preproc: variable to decide if the preprocessing data should be saved
    @rtype save_preproc: bool
    
    @return preproc_file: the file where the preprocessing data should be saved
    @rtype preproc_file: str
    
    @return save_features: variable to decide if the features should be saved
    @rtype save_features: bool
    
    @return features_file: the file where the features should be saved
    @rtype features_file: str
    
    @return preprocess_only: variable to decide if only do the extraction of preprocessing data
    @rtype preprocess_only: bool
    
    @return features_only: variable to decide if only do the extraction of features
    @rtype features_only: bool
    '''
    save_preproc = False
    save_features = False
    if config_savepreproc == True:
        save_preproc = True
    elif '-sp' in args:
        save_preproc = True  
    if config_savefeatures == True:
        save_features = True
    elif '-sf' in args:
        save_features = True

    preprocess_only = False
    features_only = False
    if config_preproconly == True or '-preprocess_only' in args:
        print("Extracting preprocessing data only. No classification prediction will take place.")
        preprocess_only = True
        save_preproc = True
    elif config_featuresonly == True or '-features_only' in args:
        print("Extracting features only. No classification prediction will take place...")
        features_only = True
        save_features = True
        
    if save_preproc:
        print("Saving of preprocessing data enabled...")
        print("WARNING: depending on the number of files to process this could take a big amount of space.")
    if save_features:
        print("Saving of features values enabled...")
        
    preproc_file = None
    features_file = None
    if '-pf' in args:
        index_pp = args.index('-pf') + 1
        preproc_file = args[index_pp]
    if 'rf' in args:
        index_rp = args.index('-rf') + 1
        features_file = args[index_rp]

    return save_preproc, preproc_file, save_features, features_file, preprocess_only, features_only
    