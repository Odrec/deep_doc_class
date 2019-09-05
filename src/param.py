#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 11:09:01 2017

@author: odrec
"""

import os, csv, argparse, sys
from os.path import realpath, basename, dirname, join, splitext, isfile, isdir
from glob import glob
import logging.config
import paths

logging.config.fileConfig(fname=paths.LOG_FILE, disable_existing_loggers=False)
# Get the logger specified in the file
logger = logging.getLogger("copydocLogger")
debuglogger = logging.getLogger("debugLogger")
        
def load_config_file(config_file): 
    '''
    Loads the parameters from the configuration file
    
    @param config_file: the configuration file
    @dtype config_file: str
    
    @return config_params: all the parameters loaded from the configuration file
    @rtype config_params: dict
    '''
    configuration_list = [line.strip() for line in open(config_file).readlines()]
    configuration_list = [line.split('=') for line in configuration_list]
    config_params = {}
    for l in configuration_list:
        if l[0] == 'metadata_file':
            config_params['metadata_file'] = l[1]
        elif l[0] == 'batch':
            config_params['batch'] = l[1]
        elif l[0] == 'models_path':
            config_params['models_path'] = l[1]
        elif l[0] == 'cores':
            config_params['cores'] = l[1]
        elif l[0] == 'predict_threshold':
            config_params['threshold'] = l[1]
        elif l[0] == 'save_preprocessing':
            config_params['save_preprocessing'] = True
        elif l[0] == 'preprocessing_file':
            config_params['preprocessing_file'] = l[1]
        elif l[0] == 'features_file':
            config_params['features_file'] = l[1]
        elif l[0] == 'results_file':
            config_params['prediction_file'] = l[1]
        elif l[0] == 'train':
            config_params['train'] = l[1]
        elif l[0] == 'deep':
            config_params['deep'] = l[1]
        elif l[0] == 'load':
            config_params['load'] = l[1]

    if not config_params['save_preprocessing']: config_params['save_preprocessing'] = False
    if not config_params['train']: config_params['train'] = False
    if not config_params['deep']: config_params['deep'] = False
    if not config_params['load']: config_params['load'] = False
                
    return config_params
    
def get_metadata(meta_file, ids):
    '''
    Extracts the metadata
    
    @param/return meta_file: path to the metadata file
    @rtype meta_file: str
    '''
    doc_ids = []
    metadata = {}
    extra_info = {}
    #change this if more metadata features are added
    metadata_feature_list = ['file_name', 'folder_name']
    extra_list = ['course_name', 'number_participants']
    if meta_file and isfile(meta_file): 
        reader = csv.DictReader(open(meta_file, 'r'))
        for row in reader:
            key = row.pop('document_id')
            if key in ids:
                doc_ids.append(key)
                entrydict = {}
                for f in metadata_feature_list:
                    entrydict[f] = row[f]
                extradict = {}
                for f in extra_list:
                    if f in row: extradict[f] = row[f]
                metadata[key] = entrydict
                if extradict: extra_info[key] = extradict
    elif isinstance(meta_file, str) and len(ids) == 1: 
        doc_ids.append(ids[0])
        metadata[ids[0]] = {}
        metadata_list = meta_file.replace('=',' ').replace(',',' ').split()
        for f in metadata_feature_list:
            index_f = metadata_list.index(f)+1
            metadata[ids[0]][f] = metadata_list[index_f]
    else:
        metadata = {}
    return metadata, extra_info
    
def check_batch_value(batch_value, num_files):
    '''
    Compares the value for batch to the total number of files and returns appropiate value
    
    @param num_files: the total number of files
    @dtype num_files: int
    
    @param/return batch_value: number of batch files
    @rtype batch_value: int
    '''
    if num_files < batch_value:
        logger.info("Using total number of files as value of batch...")
        batch_value = num_files
    return batch_value
    
def check_number_of_cores(batch_quant, cores):
    '''
    Compares the number of cores to be used to the batch value and returns the appropiate value
    
    @param batch_quant: the number of files in a batch
    @dtype batch_quant: int
    
    @param/return cores: the number of cores specified
    @dtype cores: int

    '''
    if cores > batch_quant:
        logger.info("Using the number of files per batch for the number of cores...")
        cores = batch_quant
    return cores

def get_pdf_path_files(pdf_path, pdf_files=None):
    '''
    Gets all the pdf files
    
    @param pdf_path: path to the pdf files
    @dtype pdf_path: str
    
    @return pdf_files: pdf files
    @rtype pdf_files: list
    '''
    ids = []
    if not pdf_files: pdf_files = glob(join(pdf_path,"*.{}".format('pdf')))
    for f in pdf_files: ids.append(splitext(basename(f))[0])
    return pdf_files, ids
    
def is_pdf(file_path):
    '''
    Check validity of parameter for path to file(s)
    
    @param file_path: path to pdf file(s) specified
    @dtype file_path: str
    
    @return files: list of pdf file(s)
    @rtype files: list
    
    @return idf/ids: list of file id(s)
    @rtype idf/ids: list
    '''
    if isfile(file_path) and file_path.endswith('.pdf'): 
        idf = splitext(basename(file_path))[0]              
        return file_path, idf
    elif isdir(file_path):
        files, ids = get_pdf_path_files(file_path)
        if files: return files, ids
    else:
        pdf_files = file_path.split()
        for f in pdf_files:
            if isfile(f): pass
            else: 
                logger.error("No valid input file(s) were specified.")
                raise argparse.ArgumentTypeError('argument -fp or -fl must be of type *.pdf or a path to multiple pdf files.')
        files, ids = get_pdf_path_files(dirname(pdf_files), pdf_files)
        return files, ids
    raise argparse.ArgumentTypeError('argument -fp must be of type *.pdf or a path to multiple pdf files.')

def is_conf(conf_file):
    '''
    Check validity of parameter for configuration file
    
    @param/return conf_file: path to configuration file specified
    @dtype conf_file: str
    '''
    logger.info("Using config file. Parameters given in the command line will be ignored.")
    if isfile(conf_file): 
        config_data = load_config_file(conf_file)
        return config_data 
    logger.error("Invalid configuration file.")
    raise argparse.ArgumentTypeError(
            'argument -conf must be a valid configuration file.')
    
def is_meta(meta_file):
    '''
    Check validity of parameter for metadata file
    
    @param/return meta_file: path to metadata file specified
    @dtype meta_file: str
    '''
    logger.info("Using metadata file...")
    if isfile(meta_file) and meta_file.endswith('.csv'):
        return meta_file
    logger.error("Invalid metadata file.")
    raise argparse.ArgumentTypeError(
            'argument -meta must be a valid csv metadata file.') 
    
def is_batch(batch):
    '''
    Check validity of parameter for batch
    
    @param/return batch: the value of batch specified
    @dtype batch: int
    '''
    logger.info("Setting batch value...")
    if batch.isdigit():
        batch = int(batch)
        if batch > 0:
            return batch
    logger.error("Invalid value for batch.")
    raise argparse.ArgumentTypeError(
            'argument -b must be a valid positive value for batch.') 

def is_cores(cores):
    '''
    Check validity of parameter for number of cores
    
    @param/return cores: the number of cores specified
    @dtype cores: int

    '''
    logging.info("Setting cores value...")
    cpus = os.cpu_count()
    if cores.isdigit():
        cores = int(cores)
        if cores > cpus:
            logger.warning("Using the maximum number of cores available. \
                           This could limit the performance of your computer considerably.")
            return None
        if cores > 0:
            return cores
    logger.error("Invalid value for number of cores.")
    raise argparse.ArgumentTypeError(
            'argument -c must be a valid positive value of cores.') 
    
def is_threshold(threshold):
    '''
    Check validity of parameter for threshold
    
    @param/return threshold: the threshold value specified
    @dtype threshold: float

    '''
    logger.info("Setting threshold value...")
    try:
        threshold = float(threshold)
        if threshold > 0 and threshold < 1:
            return threshold
    except:
        logger.error("Invalid threshold value.")
    raise argparse.ArgumentTypeError(
        'argument -t must be a valid value between 0 and 1.') 
    
def is_preprocessing(preprocessing_file):
    '''
    Checks and creates folders and files necessary for storing preprocessing data

    @param/return preprocessing_file: preprocessing file specified
    @dtype preprocessing_file: str
    '''
    logger.info("Using file to store preprocessing data...")
    logger.info("Saving of preprocessing data enabled...")
    logger.warning("Depending on the number of files to process this could take a big amount of space.")
    if isfile(preprocessing_file) and preprocessing_file.endswith('.json'):
        return preprocessing_file
    preprocessing_path = paths.PREPROCESSING_PATH
    text_path = paths.TEXTFILES_PATH
    try:
        if not isdir(preprocessing_path):
            os.makedirs(preprocessing_path)
            os.makedirs(text_path)
        if not isdir(text_path):
            os.makedirs(text_path)
        return preprocessing_file
    except Exception as e:
        logger.error("Error while creating folders and files for preprocessing. %s"%e)
    raise argparse.ArgumentTypeError(
            'argument -sp must be a json file.') 
    
def is_features(features_file):
    '''
    Checks and creates folders and files necessary for storing feature data
    
    @param/return features_file: features file specified
    @dtype features_file: str
    '''
    logger.info("Using file to store features data...")
    logger.info("Saving of features data enabled...")
    if isfile(features_file) and features_file.endswith('.json'):
        return features_file
    elif features_file.endswith('.json'):
        data_path = join(join(dirname(realpath(__file__)), os.pardir), "data")
        preprocessing_path = join(data_path, "preprocessing_data")
        text_path = join(preprocessing_path, "text_files")
        try:
            if not isdir(preprocessing_path):
                os.makedirs(preprocessing_path)
                os.makedirs(text_path)
            if not isdir(text_path):
                os.makedirs(text_path)
            features_file = join(preprocessing_path, features_file)
            open(features_file, 'w').close()
            return features_file
        except:
            logger.error("Invalid features file.")
    raise argparse.ArgumentTypeError(
            'argument -ff must be a valid json file.') 
    
def is_training(labels_file):
    '''
    Checks and loads label data
    
    @param labels_file: labels file specified
    @dtype labels_file: str
    @return labels: labels for all training documents
    @dtype labels: list
    '''
    logger.info("Checking and reading labels file...")
    labels_dict = {}
    if isfile(labels_file) and labels_file.endswith('.csv'):
        with open(labels_file, "r") as labelsFile:
            labels_reader = csv.reader(labelsFile)
            for row in labels_reader:
                labels_dict[row[1]] = float(row[0])
        return labels_dict
    else:
        debuglogger.error("Training file with labels %s not found."%(labels_file))
        logger.warning("No labels file found!")
    raise argparse.ArgumentTypeError(
            'argument -train must come with a valid and existing csv file with label data.')
    
def is_results(results_path):
    '''
    Checks and creates folders necessary for storing results
    
    @param/return results_path: path to where the results will be saved
    @dtype results_path: str
    '''
    logger.info("Using path %s to store results..."%results_path)
    if not isdir(results_path):
        os.makedirs(results_path)
    return results_path
    raise argparse.ArgumentTypeError(
            'argument -rp must be a valid directory or be in a path where a directory can be created.') 
    
def is_mod(models_path):
    '''
    Check validity of parameter for the path to the models
    
    @param/return models_path: path to the trained models specified
    @dtype models_path: str

    '''
    logger.info("Using trained models...")
    if isdir(models_path):
        return models_path
    logger.error("Invalid path to trained models.")
    raise argparse.ArgumentTypeError(
            'argument -mod must be a valid path to the trained models.')
    
def is_loadclassified(processed_docs_file):
    '''
    Checks the file for processed files and loads the file ids on a list
    
    @return files_to_avoid: ids of files that were already processed
    @dtype files_to_avoid: list

    '''
    logger.info("Loading the ids of files that were already processed...")
    files_to_avoid = []
    if isfile(processed_docs_file):
        with open(processed_docs_file, 'r') as f:
            reader = csv.reader(f)
            files_to_avoid = list(reader)[0]
    return files_to_avoid
    raise argparse.ArgumentTypeError('Error ocurred while trying to load the ids of processed files with the -loadclassified parameter.')

def process_params(args):
    '''
    Process the parameters
    
    @param args: list of arguments passed as parameters
    @dtype args: list
    
    @return params: dictionary of parameters
    @rtype params: dict
    '''    
    parser = argparse.ArgumentParser(description='Copyright document classification software.')
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    
    #Check for required path to files or file
    required.add_argument('-fp', help='path to pdf file(s) or list of pdf files. If you use saved features data this is not required, otherwise it is required.', type=is_pdf)
    optional.add_argument('-fl', nargs='+', default=[], help='list of pdf files to process.', type=is_pdf)

    #Checks if there's a metadata file specified.
    optional.add_argument('-meta', nargs='?', help='specifies metadata file and whether to use metadata for classification.', type=is_meta)
    #Check if number of cores is specified. Default value 1.
    optional.add_argument('-c', default=1, help='specifies amount of cores for parallel processing.', type=is_cores)
    #Checks if there's a configuration file specified. Takes precedence over command line parameters
    optional.add_argument('-conf', help='specifies configuration file.', type=is_conf)
    #Check if preprocessing file is specified.
    optional.add_argument('-pf', const=paths.FEATURES_FILE, nargs='?', help='specifies the name for the file to load the features data. If -fp is also used then this flag and the data to load will be ignored.')
    #Check if save preprocess data is specified, only the preprocessing will be done
    optional.add_argument('-po', const=paths.FEATURES_FILE, nargs='?', help='specifies that the users wants to only preprocess data. The preprocess data will be saved.', type=is_preprocessing)
    #File with the training data (document ids and labels)
    optional.add_argument('-train', const=paths.LABELS_FILE, nargs='?', help='specifies if the user wants to train the classification system and load the label data. You can pass the labels file.', type=is_training)
    #Checks if deep learning models should be used
    optional.add_argument('-deep', nargs='?', const=True, help='specifies the path to the unlabeled image data needed for the training procedure.\
                          If specified without a path, then it is used during classification to use the trained deep models.WARNING: While in training mode this can take a huge amount of time and space.')
    optional.add_argument('-overwrite', action='store_true', help='will overwrite all saved data, if any. If not specified, the program will try to concatenate the data to existing files.')  
    optional.add_argument('-report', action='store_true', help='Generate a report with the results and other helpful statistics.')  
    optional.add_argument('-manual', action='store_true', help='Provides a random sample of positively classified documents for manual evaluation.') 
    optional.add_argument('-sample', const=100, nargs='?', help='Process just a random sample of the documents. Default value is 100.')
    optional.add_argument('-load_classified', const='../data/processed_files.csv', nargs='?', help='Checks which files are already classified by checking the file ../data/processed_files.csv and takes them out of the list from files to process.', type=is_loadclassified)
    
    #PARAMETERS ONLY FOR NOT TRAINING MODE
    #Check if results path is specified.
    optional.add_argument('-rp', const=paths.RESULTS_PATH, nargs='?', help='Specifies the path to store the results, report and sample for manual evaluation.', type=is_results)
    #Check if batch is specified. Default value 1.
    optional.add_argument('-b', default=1, help='ONLY USED IF NOT ON TRAINING MODE. Specifies amount of files per batch.', type=is_batch)
    #Check if threshold is specified. Deafult value 0.5.
    optional.add_argument('-t', default=0.5, help='ONLY USED IF NOT ON TRAINING MODE. Specifies the value for the threshold for the classification decision. The results will be shown in the results file.', type=is_threshold)
    #Checks if there's a path to the trained models specified.
    optional.add_argument('-mod', const=paths.MODELS_PATH, nargs='?', help='ONLY USED IF NOT ON TRAINING MODE. Specifies path to trained models. If not specified the default path (models/) will be used.', type=is_mod)
    
    args = parser.parse_args()
    params = {}
    debuglogger.debug("Setting up parameters values.")
    params['pdf_files'] = None
    params['ids'] = None
    params['features_file'] = None
    params['load'] = False
    params['train'] = False
    params['overwrite'] = args.overwrite
    params['metadata'] = None
    params['extra'] = None
    params['sample'] = False
    if not args.rp: params['results_path'] = paths.RESULTS_PATH
    else: params['results_path'] = args.rp
    if args.pf:
        params['load'] = True
        params['features_file'] = args.pf
        if not isfile(params['features_file']):
            logger.error("The file %s with the data to load doesn't exist!"%params['features_file'])
            debuglogger.error("The file %s with the data to load doesn't exist!"%params['features_file'])
            sys.exit(1)
        if args.meta: params['metadata'] = True
    elif args.fp or args.fl:
        if args.fp:
            params['pdf_files'] = args.fp[0]
            params['ids'] = args.fp[1]
        else:
            files_data = args.fl
            params['pdf_files'] = []
            params['ids'] = []
            for i,f in enumerate(files_data):
                params['pdf_files'].append(args.fl[i][0])
                params['ids'].append(args.fl[i][1])
        if args.sample:
            params['sample'] = True
            value_sample = int(args.sample)
            if value_sample < len(params['pdf_files']):
                import random
                tmp = random.sample(params['pdf_files'], value_sample)
                pdf_indexes = [i for i in range(len(params['pdf_files'])) if params['pdf_files'][i] in tmp]
                params['ids'] = [params['ids'][i] for i in pdf_indexes]
                #rearrange pdf files because the sample took them without order
                params['pdf_files'] = [params['pdf_files'][i] for i in pdf_indexes]
        elif args.load_classified:
            files_to_take_out = args.load_classified
            indexes_to_delete = []
            for i,f in enumerate(files_to_take_out):
                for j,pdf in enumerate(params['ids']):
                    if files_to_take_out[i] == params['ids'][j]:
                        indexes_to_delete.append(j)
            for i in sorted(indexes_to_delete, reverse=True):
                params['pdf_files'].pop(i)
                params['ids'].pop(i)
        params['num_files'] = len(params['ids'])
        if params['num_files'] == 0:
            logger.info("There are no new files to process. Exiting and going to next batch.")
            debuglogger.info("There are no new files to process. Exiting and going to next batch.")
            sys.exit(1)
        #Get the data from the metadata file
        if args.meta: 
            metadata_file = args.meta
            params['metadata'], params['extra'] = get_metadata(metadata_file, params['ids'])
    if not params['pdf_files'] and not params['features_file']:
        logger.error("You need to specify a path to the pdf file(s) to process with the -fp argument or specify a features json file with the saved features with the parameter -pf.")
        debuglogger.error("You need to specify a path to the pdf file(s) to process with the -fp argument or specify a features json file with the saved features with the parameter -pf.")
        sys.exit(1)
    params['save_preprocessing'] = False
    if args.po:
        params['save_preprocessing'] = True
        if params['train']: logger.info("Extracting and saving preprocessing data only. No training will take place.")
        else: logger.info("Extracting and saving preprocessing data only. No classification prediction will take place.")
        if not params['features_file']: 
            params['features_file'] = args.po  
    params['labels'] = None
    params['deep'] = False
    params['unlabeled_pdfs_path'] = None
    params['cores'] = 1
    params['batch'] = 1
    params['report'] = args.report
    params['manual'] = args.manual
    params['threshold'] = 0.5
    params['models_path'] = paths.MODELS_PATH
    if args.conf:
        metadata_file = args.conf[0]
        params['cores'] = args.conf[3]
        params['features_file'] = args.conf[10]
        params['save_preprocessing'] = args.conf[8]
        params['batch'] = args.conf[1]
        params['models_path'] = args.conf[2]
        params['threshold'] = args.conf[4]
        params['train'] = args.conf[12]
        params['deep'] = args.conf[13]
    else:
        if args.train: 
            params['train'] = True
            if not params['load']:
                params['labels'] = args.train
                all_labels = []
                if params['labels']:
                    for f in params['pdf_files']:
                        i_d = basename(f).split('.')[0]
                        if i_d in params['labels']: all_labels.append(params['labels'][i_d])
                        else: 
                            debuglogger.error("No label for file %s. Training will not continue."%(f))
                            logger.error("No label for file %s. Training will not continue."%(f))
                            sys.exit(1)
                            
                    params['labels'] = all_labels
        if args.deep: 
            params['deep'] = True
            if params['train']:
                params['unlabeled_pdfs_path'] = args.deep
                if not isdir(params['unlabeled_pdfs_path']): 
                    debuglogger.debug("No path to unlabeled files was specified. The training will be done without pseudo-labeling.")
                    params['unlabeled_pdfs_path'] = None
        if args.c: params['cores'] = args.c
        if not params['train']:
            if args.b: params['batch'] = args.b #FOR NOW LET'S KEEP BATCH AT MAXIMUM
            if args.t: params['threshold'] = args.t
            if args.mod: params['models_path'] = args.mod
        else:
            logger.info("When training using all files as batch.")
            #Sometimes num_files doesn't exist because the data is going to be loaded
            if 'num_files' in params: params['batch'] = params['num_files']
            
    if not params['train']:
        #Check if batch value is not higher than number of files.
        #Sometimes num_files doesn't exist because the data is going to be loaded
        if 'num_files' in params: params['batch'] = check_batch_value(params['batch'], params['num_files'])
        #Check if cores value is valid.
        params['cores'] = check_number_of_cores(params['batch'], params['cores'])
    return params