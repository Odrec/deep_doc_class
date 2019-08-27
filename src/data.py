#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 16:57:01 2017

@author: odrec
"""

import json, os, random, sys
import numpy as np
from os.path import basename, join, splitext, dirname, realpath, isfile, isdir
import csv
import pandas as pd
from glob import glob
import logging.config
from sklearn.utils import shuffle

from Feature_Extractor import FeatureExtractor
import features.pdf_properties as fp
import features.pdf_xml_structure_new as fxs
from bow_classifier.bow_classifier import BowClassifier
import features_names
#Adds top directory to path to import pdftojpg
sys.path.append("..")
from help_scripts import pdftojpg

import paths
logging.config.fileConfig(fname=paths.LOG_FILE, disable_existing_loggers=False)
# Get the logger specified in the file
logger = logging.getLogger("copydocLogger")
debuglogger = logging.getLogger("debugLogger")

MEANS_FILE = join(join(dirname(realpath(__file__)), os.pardir), "data/features_means/means.csv")

def get_metadata(batch_files, metadata):
    '''
    Gets the metadata for the list of batch files
    
    @param batch_files: The list of batch files
    @dtype batch_files: list

    @param metadata: the metadata loaded from file
    @dtype metadata: dict

    @return batch_meta: The metadata for the batch files
    @rtype  batch_meta: dict
    '''
    batch_meta = {}
    for i, f in enumerate(batch_files):
        doc_id = splitext(basename(f))[0]
        if doc_id in metadata.keys():
            debuglogger.debug("Getting metadata for file with id: %s",doc_id)
            batch_meta[doc_id] = metadata[doc_id]
        else:
            logger.warning("No metadata found for file with id: %s",doc_id)
            batch_meta[doc_id] = {}
            batch_meta[doc_id]['filename'] = " " 
            batch_meta[doc_id]['folder_name'] = " "
    return batch_meta   
 
def get_preprocessing_data(batch_files, metadata, cores):
    '''
    Gets the preprocessing data for the list of batch files
    
    @param batch_files: The list of batch files
    @dtype batch_files: list

    @param metadata: the metadata loaded from file
    @dtype metadata: dict
    
    @param cores: the number of cores to be used for extracting the preprocessing data
    @dtype cores: int

    @return ids: list of document ids of the batch
    @rtype ids: list
    
    @return preprocessing_data: list of dicts with all the preprocessing data
    @rtype preprocessing data: list
    '''
    batch_meta = {}
    if metadata:
        logger.info("Starting preprocessing metadata.")
        batch_meta = get_metadata(batch_files, metadata)
        logger.info("Finished preprocessing metadata.")
        
    logger.info("Starting preprocessing properties.")
    properties = fp.pre_extract_pdf_properties(batch_files, cores)
    logger.info("Finished preprocessing properties.")

    logger.info("Starting preprocessing structure.")
    structure = fxs.pre_extract_pdf_structure_data(batch_files, cores)
    logger.info("Finished preprocessing structure.")
    preprocessing_data = [batch_meta, properties, structure]
    return preprocessing_data

def save_preproc_data(doc_ids, preprocessing_list, preprocessing_file='../data/preprocessing_data/preprocessing_data.json'):
    '''
    Saves the preprocessing data
    
    @param doc_ids: The list of doc ids to save
    @dtype doc_ids: list

    @param preprocessing_list: list of dicts with all the preprocessing data
    @dtype preprocessing_list: list
    
    @param preproc_file: the file on which to store the preprocessing data
    @dtype preproc_file: str
    
    @return err: specifies if there was an error reading or opening the file
    @rtype err: bool
    '''
    preprocessing_path = dirname(preprocessing_file)
                
    preproc_data = {}
    metadata = preprocessing_list[0]
    properties = preprocessing_list[1]
    structure = preprocessing_list[2]
    for d in doc_ids:
        preproc_data[d] = {}
        if metadata:
            preproc_data[d]['metadata'] = metadata[d]
        if properties:
            preproc_data[d]['properties'] = properties[d]
        if structure:
            preproc_data[d]['structure'] = structure[d]
        if 'text' in list(preproc_data[d]['structure'].keys()):
            text_path = join(preprocessing_path,'text_files')
            text_file = join(text_path, d + '.txt')
            os.makedirs(dirname(text_file), exist_ok=True)
            with open(text_file, "w+") as text_file:
                text_file.write(str(preproc_data[d]['structure'].pop('text', None)))
    try:
        with open(preprocessing_file, "r") as jsonFile:
            data = json.load(jsonFile)
        data.update(preproc_data)
    except:
        data = preproc_data
    try:
        os.makedirs(dirname(preprocessing_file), exist_ok=True)
        with open(preprocessing_file, "w") as jsonFile:
            json.dump(data, jsonFile)
        return True
    except Exception as e: 
        return e
        
def get_preprocessing_dictionary(doc_ids, preprocessing_list, metadata):
    preproc_dict = {}
    for did in doc_ids:
        preproc_dict[did] = {}
        if metadata: preproc_dict[did].update(preprocessing_list[0][did])
        preproc_dict[did].update(preprocessing_list[1][did])
        preproc_dict[did].update(preprocessing_list[2][did])
    return preproc_dict    
        
def get_features(doc_ids, preprocessing_list, metadata, cores, models_path, train=False):
    '''
    Extracts the features
    
    @param doc_ids: The list of doc ids to extract features from
    @dtype doc_ids: list

    @param preprocessing_list: list of dicts with all the preprocessing data
    @dtype preprocessing_list: list
    
    @param metadata: metadata for the doc_ids
    @dtype metadata: str
    
    @param train: whether the request for the extraction of features comes from training or not
    @dtype train: bool
    
    @return features: dictionary with all the extracted features
    @rtype: dict
    '''

    fe = FeatureExtractor(doc_ids, metadata, preprocessing_list[1], preprocessing_list[2], train)
    fe.get_bow_features(models_path)
    count_pages = fe.get_numeric_features()
    features = fe.feature_values
    debuglogger.debug("Created a features array with shapes. Number of features: %s", str(len(features)))
    for i,f in enumerate(features):
        debuglogger.debug("Shape of feature number %s: %s"%(str(1),str(features[i].shape)))
    return features, count_pages
    
def save_features(numeric_features, preprocessing_list, bow_features_names,\
                  numeric_features_names, batch_files, batch_ids, batch_labels,\
                  features_file, count_pages=[], overwrite=False):
    '''
    Save the features
    
    @param numeric_features: array with the numeric features
    @dtype numeric_features: array
    
    @param preprocessing_list: list of dicts with all the preprocessing data
    @dtype preprocessing_list: list of dicts
    
    @param bow_features_names: list of names of the bow features
    @dtype bow_features_names: list
    
    @param numeric_features_names: list of names of the numeric features
    @dtype numeric_features_names: list
    
    @param batch_files: list of batch files
    @dtype batch_files: list
    
    @param batch_ids: list of ids of the batch files
    @dtype batch_ids: list
    
    @param batch_labels: list of ids of the batch files
    @dtype batch_labels: list

    @param features_file: the file on which to store the features
    @dtype features_file: str
    
    @param overwrite: whether or not to overwrite the features file
    @dtype overwrite: bool
    
    @return err: returns whether or not there was an error reading or opening the file and if there was returns the error
    @rtype err: bool/object
    '''
    features_dict = {}
    text_path = join(dirname(features_file),'text_files')
    if isdir(text_path):
        if overwrite:
            files = glob(text_path+'*.txt')
            for f in files: os.remove(f)
    else: os.makedirs(text_path)
    for i,did in enumerate(batch_ids):
        features_dict[did] = {}
        if batch_labels: features_dict[did]['label'] = batch_labels[i]
        for bf in bow_features_names:
            if preprocessing_list[0] and bf in preprocessing_list[0][did].keys():
                features_dict[did][bf] = preprocessing_list[0][did][bf]
            elif preprocessing_list[1] and bf in preprocessing_list[1][did].keys():
                features_dict[did][bf] = preprocessing_list[1][did][bf]
            elif preprocessing_list[2] and bf in preprocessing_list[2][did].keys():
                #save text apart
                if bf == "text":
                    text_file = join(text_path, did + '.txt')
                    with open(text_file, "w+") as text_file:
                        text_file.write(str(preprocessing_list[2][did]['text']))
                else: features_dict[did][bf] = preprocessing_list[2][did][bf]
        for j,nf in enumerate(numeric_features_names):
            features_dict[did][nf] = numeric_features[i][j] 
        features_dict[did]['number_pages'] = count_pages[i]

    if overwrite:
        with open(features_file, "w") as jsonFile:
            json.dump(features_dict, jsonFile)
        overwrite = False
    else:
        try:
            with open(features_file, "r") as jsonFile:
                data = json.load(jsonFile)
            data.update(features_dict)
            features_dict = data
        except Exception as e:
            debuglogger.error("Something went wrong opening the file and saving the data.\
                              The file will be newly created: %s. Error: %s"%(features_file,e))
        try:
            with open(features_file, "w") as jsonFile:
                json.dump(features_dict, jsonFile)
        except Exception as e:
            return e, preprocessing_list, batch_files, batch_ids, batch_labels        
    return False, overwrite

    
def load_features(models_path, features_file, metadata, batch_ids=None, batch_labels=None, save=False, train=False):
    '''
    Load the features
    
    @param models_path: path to where the vectorizer models are located to generate the BOW features
    @dtype models_path: str

    @param features_file: the file on which to store the features
    @dtype features_file: str
    
    @param metadata: metadata info
    @dtype metadata: list/None
    
    @return batch_ids: list of documents ids
    @rtype batch_ids: list
    
    @return batch_labels: list of documents labels
    @rtype batch_labels: list
    
    @return remaining_ids: list of documents ids that need to still to be processed
    @rtype remaining_ids: list
    
    @return remaining_labels: list of documents labels that need to still to be processed
    @rtype remaining_labels: list
    '''
    #variables for saving process
    remaining_ids = []
    remaining_labels = []
    remaining_indexes = []
    failed_ids = []
    ##############################
    feature_data = None
    doc_ids = []
    doc_labels = []
    count_pages = []
    if isfile(features_file):        
        feature_data = []
        bow_text_features_names = features_names.bow_text_features
        bow_prop_features_names = features_names.bow_prop_features
        if metadata: bow_meta_features_names = features_names.bow_meta_features
        else: bow_meta_features_names = []
        bow_features_names = bow_text_features_names + bow_prop_features_names + bow_meta_features_names 
        numeric_features_names = features_names.numeric_features 
        txt_files = glob(join(join(dirname(features_file),'text_files'),"*.{}".format('txt')))
        text_data = []
        doc_ids = []
        doc_labels = []
        for tf in txt_files:
            doc_ids.append(splitext(basename(tf))[0])
            with open(tf, "r+") as text_file:
                text_data.append(text_file.read())
                                
        with open(features_file, "r") as jsonFile:
            json_data = json.load(jsonFile)        
        BC = BowClassifier()
        bow_data = {}
        for i,did in enumerate(doc_ids):
            bow_data[did] = {}
            for bf in bow_features_names:
                if bf == "text": bow_data[did][bf] = text_data[i]
                else: 
                    try:
                        bow_data[did][bf] = json_data[did][bf]
                        debuglogger.info("Data for feature %s on file with id %s was loaded!"%(bf, did))
                    except:
                        #Only for saving process. If there's an error getting 
                        #any saved value then calculate values again
                        if save and batch_ids and (did in batch_ids):
                            failed_ids.append(did)
                            debuglogger.error("No data for feature %s on file with id %s could be loaded!"%(bf, did))
                            debuglogger.info("Data for file with id %s will be extracted."%(did))
                            break
        if not save:
            for bf in bow_features_names:
                if bf in bow_text_features_names: origin_data = "text"
                elif bf in bow_prop_features_names: origin_data = "prop"
                elif bf in bow_meta_features_names: origin_data = "meta"
                BC.load_vectorizer_model(bf, models_path)
                feature_data.append(BC.get_vectorizer_output(doc_ids, bow_data, bf, models_path, origin_data, metadata))
           
        if not save:
            tmp_array = np.zeros(shape=(len(doc_ids),len(numeric_features_names)), dtype=float)
            for i,did in enumerate(doc_ids):
                if train:
                    try:
                        doc_labels.append(json_data[did]['label'])
                    except: 
                        debuglogger.error("The saved data should come with labels if using it for training!")
                        logger.error("The saved data should come with labels if using it for training!")
                        sys.exit(1)
                for j,nf in enumerate(numeric_features_names):
                    tmp_array[i][j] = json_data[did][nf]
                if 'number_pages' in json_data[did]:
                    count_pages.append(json_data[did]['number_pages'])
            tmp_array = np.nan_to_num(tmp_array) #this shouldn't be needed if normalization is correct
            feature_data.append(tmp_array)
        else:
            #Data needed for the saving process
            remaining_ids = list(set(batch_ids) - set(doc_ids))
            #extend with failed ids
            remaining_ids.extend(failed_ids)
            if remaining_ids:
                #get rid of duplicates
                remaining_ids = list(dict.fromkeys(remaining_ids))
                for idx in remaining_ids:
                    if idx in batch_ids:
                        index = batch_ids.index(idx)
                        remaining_indexes.append(index)
                        #If there are labels then data is for training
                        if batch_labels:
                            remaining_labels.append(batch_labels[index])
        ####################################
    else:
        logger.info("No file to load data from. %s"%features_file)
        debuglogger.info("No file to load data from. %s"%features_file)
    return feature_data, doc_ids, doc_labels, count_pages, remaining_ids, remaining_labels, remaining_indexes
    
def normalize_features(features, features_names, amount_of_values, train=True):
    '''
    Normalizes features
    
    @param features: matrix with the features
    @dtype features: list of arrays
    @param features_names: names of the numeric features
    @dtype features_names: list
    @amount_of_values: amount of values that have been processed
    @dtype amount_of_values: int
    @param train: whether it is in training mode or not
    @dtype train: bool
    @return features: matrix with the normalized features
    @rtype features: list of arrays
    '''
    debuglogger.debug("Normalizing features.")
    mean_values, min_values, max_values = check_means_file(features, features_names, amount_of_values, train)
    for i,doc in enumerate(features):
        for j,value in enumerate(doc):
            if value == np.NaN: 
                debuglogger.debug("NaN value on feature %s, replacing it with mean value.",str(j)) 
                features[i][j] = mean_values[j]
            else:
                f_range = (np.float64(max_values[j]) - np.float64(min_values[j]))
                if f_range > 0: 
                    features[i][j] = (value - np.float64(min_values[j])) / f_range 
                else:
                    debuglogger.debug("The max value for feature %s is not greater\
                                      than its min value. Leaving value as it is.",str(j))
    return features
    
def check_means_file(features, features_names, amount_of_values, train):
    '''
    Checks if the means.csv file exists. 
    Adjusts it to new values during training or creates the file in case it doesn't exist.
    If it doesn't exist during prediction then the application will stop and an error will be logged.
    
    @param features: matrix with the features
    @dtype features: list of arrays
    
    @param features_names: names of the numeric features
    @dtype features_names: list
    
    @param amount_of_values: amount of values that have been processed
    @dtype amount_of_values: int
    
    @param train: whether it is in training mode or not
    @dtype train: bool
    
    @return mean_values: list with the means of all features
    @rtype mean_values: list
    
    @return min_values: list with the minimum values for all features
    @rtype min_values: list
    
    @return max_values: list with the maximum values of all features
    @rtype max_values: list
    '''
    
    debuglogger.debug("Check file and if the max and min values for the features are stored.")
    debuglogger.debug("If not and in training mode, save the max and min values in "+MEANS_FILE+".")
    
    num_of_features = len(features_names)
    error_in_file = False
    file_values = None
    
    debuglogger.debug("Checking if "+MEANS_FILE+" file exists and that data is correct.")
    if isfile(MEANS_FILE): 
        debuglogger.debug("Means file exists in "+MEANS_FILE+".")
        with open(MEANS_FILE, "r") as meansFile:
            means_reader = csv.reader(meansFile)
            file_features_names = []
            #values for the rows
            file_values = [None] * 3
            #means
            file_values[0] = [None] * num_of_features
            #min values
            file_values[1] = [None] * num_of_features
            #max values
            file_values[2] = [None] * num_of_features
            #Check if the means file has 4 rows exactly
            if not check_rows_and_features(means_reader, num_of_features):
                debuglogger.error("File "+MEANS_FILE+" is not well-formed.")
                error_in_file = True
            else:
                meansFile.seek(0)            
                line_count = 0
                for row in means_reader:
                    if line_count == 0 and row:
                        for n in row:
                            if not n in features_names:
                                debuglogger.error("At least one feature's values are missing for normalization.\
                                                  Creating new means file with new values.")
                                #use the default sequence of names to create the file
                                file_features_names = features_names
                                error_in_file = True
                                break
                            else:
                                #we need the order of the feature names in the file
                                #in case they don't match the order in the features_names list
                                file_features_names.append(n)
                        line_count += 1
                    elif row: #check row is not empty
                        for i,n in enumerate(row):
                            try:
                                file_values[line_count-1][i] = float(n)
                            except Exception as e:
                                debuglogger.error("Some error ocurred while checking the values for\
                                                  feature normalization. Creating new means file with new values.\
                                                  Exception %s", e)
                                file_values = None
                                error_in_file = True
                                break
                        line_count += 1
                    else: 
                        file_values = None
                        error_in_file = True
    else:
        debuglogger.debug("Means file was not found in "+MEANS_FILE+".")
        error_in_file = True

    if train:
        debuglogger.debug("Preparing all the values for the means.csv file.")
        #Check for min and max values but only in training
        min_values = [None] * num_of_features
        max_values = [None] * num_of_features
        mean_values = [None] * num_of_features
        
        #get the value of that feature for every file and check if it's higher or lower 
        #than the current value max, min value. If it is then we need to regenerate 
        #the means file
        sum_values = [0] * num_of_features
        new_means = [0] * num_of_features
        amount_of_batch = len(features)
        for i,f in enumerate(features):
            for j,v in enumerate(f):
                try:
                    sum_values[j] += features[i][j]
                except:
                    debuglogger.error("Bad value for feature at position %s %s"%(i, j))
                if min_values[j] == None or features[i][j] < min_values[j]:
                    min_values[j] = features[i][j]
                if max_values[j] == None or features[i][j] > max_values[j]:
                    max_values[j] = features[i][j]
        for s in range(len(sum_values)):
            new_means[s] = sum_values[s] / amount_of_batch
            if file_values == None: mean_values[s] = new_means[s]
            else:
                for a,value_me in enumerate(file_values[0]):
                    #get the name of the current feature
                    current_feature_name = file_features_names[a]
                    #get the index of this feature in the features_names list
                    index_feature = features_names.index(current_feature_name)
                    #New means with the formula X_c = (m*x_a+n*x_b)/m+n
                    #m is number items on first set
                    #n is number of items on second set
                    #x_a is mean of first set
                    #x_b is mean of second set
                    mean_values[index_feature] = ((amount_of_values * file_values[0][a]) +\
                               (amount_of_batch * new_means[index_feature]))/\
                               (amount_of_values + amount_of_batch)
                               
                    if min_values[index_feature] > file_values[1][a]:
                        min_values[index_feature] = file_values[1][a]
                    if max_values[index_feature] < file_values[2][a]:
                        max_values[index_feature] = file_values[2][a]                 

        debuglogger.debug("Creating or updating means file in "+MEANS_FILE+".")
        MEANS_PATH = dirname(MEANS_FILE)
        if not isdir(MEANS_PATH): os.makedirs(MEANS_PATH)
        with open(MEANS_FILE, "w+") as meansFile:
            means_writer = csv.writer(meansFile)
            means_writer.writerow(features_names)
            means_writer.writerow(mean_values)
            means_writer.writerow(min_values)
            means_writer.writerow(max_values)
                
    else:
        if error_in_file:
            debuglogger.error("Something is wrong with the file "+MEANS_FILE+".")
            debuglogger.error("Please generate one or fix it, otherwise features can't be normalized.")  
            logger.info("Something went wrong with the means.csv file. Check the errors log.")
            sys.exit(1)
        else:
            mean_values = file_values[0]
            min_values = file_values[1]
            max_values = file_values[2]
        
    return mean_values, min_values, max_values

def check_rows_and_features(means_reader, num_of_features):
    rows = 0
    for row in means_reader:
        if len(row) == num_of_features:
            rows += 1
        else: return False
    if rows == 4: return True
    else: return False
    
def get_unlabeled_files(unlabeled_pdfs_path):
    unlabeled_pdf_files = glob(join(unlabeled_pdfs_path,"*.{}".format('pdf')))
    if unlabeled_pdf_files: return unlabeled_pdf_files
    else:
        logger.warning("No unlabeled pdf files found.")
        debuglogger.debug("No unlabeled pdf files found in %s."%(unlabeled_pdfs_path))
    
def split_data_for_training(number_of_files, features, ids, labels):
    #Get 80% of files for training and 20% for testing
    quant_train_data = int(number_of_files * 0.8)
    quant_test_data = number_of_files - quant_train_data
    logger.info("Using 80% of documents ("+str(quant_train_data)+") \
                for training and 20% of documents ("+str(quant_test_data)+") for testing.")
    td = random.sample(list(enumerate(ids)), quant_train_data)
    all_indexes = list(range(number_of_files))
    training_indexes = []
    training_ids = []
    for idx, i_d in td:
        training_indexes.append(idx)
        training_ids.append(i_d)
    testing_ids = [i_d for i_d in ids if i_d not in training_ids]
    testing_indexes = [idx for idx in all_indexes if idx not in training_indexes]
    training_features = []
    training_labels = []
    training_ids = []
    testing_features = []
    testing_labels = []
    testing_ids = []
    for i,f in enumerate(features):
        for a,idx in enumerate(training_indexes):
            if i == 0:
                training_ids.append(ids[idx])
                training_labels.append(labels[idx])
            if a == 0: tmp_array = np.zeros(shape=(len(training_indexes),features[i].shape[1]), dtype=float)
            for j,n in enumerate(features[i][idx]):
                tmp_array[a][j] = np.float64(features[i][idx][j])
        training_features.append(tmp_array)
        for a,idx in enumerate(testing_indexes):
            if i == 0:
                testing_ids.append(ids[idx])
                testing_labels.append(labels[idx])
            if a == 0: tmp_array = np.zeros(shape=(len(testing_indexes),features[i].shape[1]), dtype=float)
            for j,n in enumerate(features[i][idx]):
                tmp_array[a][j] = np.float64(features[i][idx][j])
        testing_features.append(tmp_array)
    logger.info("Finishing dividing the data for training and testing.")
    return training_features, training_labels, training_ids, testing_features, testing_labels, testing_ids

def split_images_for_training(image_matrix, unlabeled_matrix, train_ids, test_ids, ids_pages, data_path, unlabeled_flag=False):
    X_images = []
    X_train = []
    X_test = []
    y_images = []
    y_train = []
    y_test = []
    train_ids_pages = []
    test_ids_pages = []
    image_matrix = np.concatenate(image_matrix, axis=1)
    im = pd.read_parquet(join(data_path,'images.parquet'))
    X_images = np.array(im['image'])
    y_images = np.array(im['label'])
    if np.isnan(y_images).any():
        logger.error("Some of the images are missing labels. Training can not proceed.")
        debuglogger.error("Some of the images are missing labels. Training can not proceed.")
        sys.exit(1)
    y_images = np.array(im['label'])
    if unlabeled_flag and unlabeled_matrix:
        unlabeled_matrix = np.concatenate(unlabeled_matrix, axis=1)
    training_matrix = []
    testing_matrix = []
    for trid in train_ids:
        indices = [i for i, x in enumerate(ids_pages) if x == trid]
        for i in indices:
            training_matrix.append(image_matrix[i])
            X_train.append(X_images[i])
            y_train.append(y_images[i])
            train_ids_pages.append(ids_pages[i])
    for teid in test_ids:
        indices = [i for i, x in enumerate(ids_pages) if x == teid]
        for i in indices:
            testing_matrix.append(image_matrix[i])
            X_test.append(X_images[i])
            y_test.append(y_images[i])
            test_ids_pages.append(ids_pages[i])
    training_matrix, X_train, y_train, train_ids_pages =\
                            shuffle(training_matrix, X_train, y_train, train_ids_pages)
    training_matrix = np.array([np.array(xi) for xi in training_matrix])
    testing_matrix = np.array([np.array(xi) for xi in testing_matrix])
    unlabeled_matrix = np.array([np.array(xi) for xi in unlabeled_matrix])    
    return training_matrix, testing_matrix, unlabeled_matrix , X_train, y_train, \
                                    X_test, y_test, train_ids_pages, test_ids_pages

def check_and_extract_images(batch_files, batch_labels, unlabeled_pdf_files, \
                             batch_ids, unlabeled_flag, overwrite=False):
    logger.warning("Checking if images need to be extracted. This may take a long time.")
    matrix_images = []
    all_pages = []
    all_unlabeled_pages = []
    unlabeled_images = None
    list_unlabeled = []
    ids_pages = []
    ids_unlabeled_pages = []
    extracted_images_number = len(batch_files)
    for a,f in enumerate(batch_files):
        debuglogger.debug("Checking if images from file %s should be extracted."%(f))
        images_paths = pdftojpg.check_images(f, overwrite)
        debuglogger.debug("Finished checking images.")
        if not images_paths:
            debuglogger.debug("Converting images from file %s."%(f))
            images_paths, f_id = pdftojpg.convert_pdf_jpg(f)
            debuglogger.debug("Finished converting images from file %s."%(f))
        else:
            debuglogger.debug("Images from file %s already exist."%(f))
            extracted_images_number -= 1
        for i,im in enumerate(images_paths):
            tmp = []
            if i == 0: all_pages.append(len(images_paths))
            tmp.append(images_paths[i])
            if batch_labels: tmp.append(batch_labels[a])
            matrix_images.append(tmp[:])
            ids_pages.append(batch_ids[a])
            debuglogger.debug("Images from file %s extracted."%(f))
    debuglogger.debug("Images extracted from %s files."%(str(extracted_images_number)))
    if unlabeled_flag and unlabeled_pdf_files:
        extracted_images_number = len(unlabeled_pdf_files)
        for f in unlabeled_pdf_files:
            debuglogger.debug("Checking if images from file %s should be extracted."%(f))
            images_paths = pdftojpg.check_images(f, overwrite)
            debuglogger.debug("Finished checking images.")
            if not images_paths:
                debuglogger.debug("Converting images from unlabeled file %s."%(f))
                images_paths, f_id = pdftojpg.convert_pdf_jpg(f)
                debuglogger.debug("Finished converting images from unlabeled file %s."%(f))
            else:
                debuglogger.debug("Images from unlabeled file %s already exist."%(f))
                extracted_images_number -= 1
            all_unlabeled_pages.append(len(images_paths))
            for i,im in enumerate(images_paths):
                list_unlabeled.append(images_paths[i])
                ids_unlabeled_pages.append(f_id)
        debuglogger.debug("Images extracted from %s unlabeled files."%(str(extracted_images_number)))
        unlabeled_images = pd.DataFrame(columns = ["image"], data = list_unlabeled)
    if batch_labels: images = pd.DataFrame(columns = ["image", "label"], data = matrix_images)
    else: images = pd.DataFrame(columns = ["image"], data = matrix_images)
    logger.info("Finished checking and extracting all images.")
    return images, unlabeled_images, all_pages, all_unlabeled_pages, ids_pages, ids_unlabeled_pages

def check_saved_image_data(data_path, batch_files, batch_ids, batch_labels, unlabeled_pdf_files, train=False):
    '''
    Checks if there is saved image data and filters the files to the ones that are not yet saved based on the ids.
    
    @param data_path: the path where the data is saved
    @dtype data_path: str
    
    @param batch_files: the files to extract image data and save
    @dtype batch_files: list
    
    @param batch_ids: the ids of the images to save
    @dtype batch_ids: list
    
    @param batch_labels: the labels of the images to save
    @dtype batch_labels: list
    
    @param unlabeled_pdf_files: the unlabeled_image_files
    @dtype unlabeled_pdf_files: list
    
    @return filtered_files: list with files that have not been saved yet
    @rtype filtered_files: list        import sys;sys.exit(1)

    
    @return filtered_ids: list with ids of the files that have not been saved yet
    @rtype filtered_ids: list
    
    @return filtered_labels: list with labels of the files that have not been saved yet
    @rtype filtered_labels: list
    
    @return unlabeled_pdf_files: list with the filtered unlabeled files that have not been saved yet
    @rtype unlabeled_pdf_files: list
    '''
    filtered_ids = batch_ids
    filtered_labels = batch_labels
    filtered_files = batch_files
    filename = join(data_path,'images.parquet')
    if isfile(filename):
        filtered_ids = []
        filtered_labels = []
        filtered_files = []
        images = pd.read_parquet(filename)  
        image_list = images['image'].tolist()
        for i,idx in enumerate(batch_ids):
            for im in image_list:
                if idx in im:
                   break
                if im == image_list[-1]: 
                    filtered_ids.append(batch_ids[i])
                    filtered_files.append(batch_files[i])
                    if train: filtered_labels.append(batch_labels[i])
    if train:
        filename_unlbl = join(data_path,'unlabeled_images.parquet')
        copy_ulbl_files = unlabeled_pdf_files.copy()
        if isfile(filename_unlbl):
            unlabeled_images = pd.read_parquet(filename_unlbl)
            unlabeled_image_list = unlabeled_images['image'].tolist()
            for upd in unlabeled_pdf_files:
                idx = splitext(basename(upd))[0]
                for jdx in unlabeled_image_list:
                    kdx = splitext(basename(jdx))[0].split('-')[0]
                    if idx in kdx:
                        index = copy_ulbl_files.index(upd)
                        copy_ulbl_files.pop(index)
                        break
        unlabeled_pdf_files = copy_ulbl_files
    return filtered_files, filtered_ids, filtered_labels, unlabeled_pdf_files   

def get_images_from_files(data_path, batch_ids, train=False):
    unlabeled_images = None
    images_new = None
    Xim1 = []
    Xim2 = [] 
    Xim3 = []
    Xu1 = []
    Xu2 = []
    Xu3 = []
    all_pages = []
    all_unlabeled_pages = []
    filename = join(data_path,'images.parquet')
    file1 = join(data_path,"doc_gap_res.parquet")
    file2 = join(data_path,"doc_gap_xce.parquet")
    file3 = join(data_path,"doc_gap_inc.parquet")
    tmp_file1 = pd.read_parquet(file1)
    tmp_file2 = pd.read_parquet(file2)
    tmp_file3 = pd.read_parquet(file3)
    ids_pages = []
    ids_unlabeled_pages = []
    if isfile(filename):
        images = pd.read_parquet(filename)        
        image_list = images['image'].tolist()
        for i,tr in enumerate(image_list):
            image_list[i] = basename(image_list[i])
            image_list[i] = image_list[i].split('-')[0]
        matrix_images = []
        Xim1_list = tmp_file1['images'][0]
        Xim2_list = tmp_file2['images'][0]
        Xim3_list = tmp_file3['images'][0]
        for i_d in batch_ids:
            indexes = [i for i, x in enumerate(image_list) if x == i_d]
            all_pages.append(len(indexes))
            for i in indexes:
                tmp = []
                tmp.append(images['image'].iloc[i])
                if train and 'label' in images: tmp.append(images['label'].iloc[i])
                elif train:
                    debuglogger.error("No labels found! For training make sure to have data with labels.")
                    logger.error("No labels found! For training make sure to have data with labels.")
                    sys.exit(1)
                matrix_images.append(tmp[:])
                Xim1.append(Xim1_list[i])
                Xim2.append(Xim2_list[i])
                Xim3.append(Xim3_list[i])
                ids_pages.append(image_list[i])
        if train: images_new = pd.DataFrame(columns = ["image", "label"], data=matrix_images)
        else: images_new = pd.DataFrame(columns = ["image"], data=matrix_images)
        Xim1 = np.array(Xim1)
        Xim2 = np.array(Xim2)
        Xim3 = np.array(Xim3)
    else: 
        debuglogger.error("No train or test names and labels file to load! %s"%(filename))
        logger.error("No saved image data to load!")
    if train:
        filename_unlbl = join(data_path,'unlabeled_images.parquet')
        if isfile(filename_unlbl):
            unlabeled_images = pd.read_parquet(filename_unlbl)
            unlabeled_list = unlabeled_images['image'].tolist()
            Xu1_list = tmp_file1['unlabeled'][0]
            Xu2_list = tmp_file2['unlabeled'][0]
            Xu3_list = tmp_file3['unlabeled'][0]
            Xu1 = np.array([np.array(xi) for xi in Xu1_list])
            Xu2 = np.array([np.array(xi) for xi in Xu2_list])
            Xu3 = np.array([np.array(xi) for xi in Xu3_list])
            tmp_name = None
            pages_counter = 0
            for i,ui in enumerate(unlabeled_list):
                name = basename(unlabeled_list[i])
                name = unlabeled_list[i].split('-')[0]
                ids_unlabeled_pages.append(name)
                if name == tmp_name:
                    pages_counter += 1
                if name != tmp_name or i == len(unlabeled_list)-1:
                    if tmp_name != None: all_unlabeled_pages.append(pages_counter)
                    tmp_name = name
                    pages_counter = 1            
        else: debuglogger.warning("No unlabeled names to load! %s"%filename_unlbl)
    return images_new, unlabeled_images, all_pages, all_unlabeled_pages, ids_pages, ids_unlabeled_pages,\
             [Xim1, Xim2, Xim3], [Xu1, Xu2, Xu3]
