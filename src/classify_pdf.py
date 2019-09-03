#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 12:04:01 2016

@author: odrec

Main script for classifying copyrighted pdf documents.
    
"""
import sys, json, csv, os
from time import time, strftime
from os.path import isfile, join, isdir, dirname
import param as pa
import data as da
import models as mo
import deep_features as df
import logging.config
import features_names
import numpy as np
from itertools import chain
from multiprocessing import Pool

import paths
logging.config.fileConfig(fname=paths.LOG_FILE, disable_existing_loggers=False)
# Get the logger specified in the file
logger = logging.getLogger("copydocLogger")
debuglogger = logging.getLogger("debugLogger")

def preprocess_training_data(features_array, num_files, ids, labels, data_path, image_matrix=None, unlabeled_matrix=None, ids_pages=None, unlabeled_flag=False):
    '''
    Get the data ready for the training process
    
    @param features_array: array with the features
    @dtype features_array: array
    
    @param num_files: Number of files
    @dtype num_files: int
    
    @param ids: The list of ids to divide into training and testing
    @dtype ids: list
    
    @param labels: The list of labels per file to divide
    @dtype labels: list
    
    @param data_path: path to where the data is stored
    @dtype data_path: str
    
    @param image_matrix: matrix with the image features
    @dtype image_matrix: matrix
    
    @param unlabeled_matrix: matrix with the unlabeled image features
    @dtype unlabeled_matrix: matrix
    
    @param ids_pages: ids of the pages for all files
    @dtype ids_pages: list
    
    @param unlabeled_flag: if unlabeled data is going to be used
    @dtype unlabeled_flag: bool
    
    @return train_feats: array with the training features
    @rtype train_feats: array
    
    @return train_lbls: array with the training labels
    @rtype train_lbls: array
    
    @return train_ids: the training ids
    @rtype train_ids: list
    
    @return test_feats: array with the testing features
    @rtype test_feats: array
    
    @return test_lbls: array with the testing labels
    @rtype test_lbls: array
    
    @return test_ids: the testing ids
    @rtype test_ids: list
    
    @return training_matrix: matrix with the training image features
    @rtype training_matrix: matrix
    
    @return testing_matrix: matrix with the testing image features
    @rtype testing_matrix: matrix
    
    @return unlabeled_matrix: matrix with the unlabeled image features
    @rtype unlabeled_matrix: matrix
    
    @return X_train: the training images paths
    @rtype X_train: list
    
    @return y_train: the training images labels
    @rtype y_train: list
    
    @return X_test: the testing images paths
    @rtype X_test: list
    
    @return y_test: the testing images labels
    @rtype y_test: list
    
    @return train_ids_pages: the training ids of the images
    @rtype train_ids_pages: list
    
    @return test_ids_pages: the testing ids of the images
    @rtype test_ids_pages: list
    '''
    logger.info("Splitting data into training and testing sets.")
    train_feats, train_lbls, train_ids, test_feats, test_lbls, test_ids = da.split_data_for_training(num_files, features_array, ids, labels)
    training_matrix = None
    testing_matrix = None
    X_train = None
    y_train = None
    X_test = None
    y_test = None
    train_ids_pages = None
    test_ids_pages = None
    if image_matrix:
        training_matrix, testing_matrix, unlabeled_matrix, X_train, y_train, X_test, y_test, train_ids_pages, test_ids_pages = da.split_images_for_training(image_matrix, unlabeled_matrix, train_ids, \
                                                                                                                                                             test_ids, ids_pages, \
                                                                                                                                                             data_path, unlabeled_flag)
    logger.info("Finished splitting data.")
    return train_feats, train_lbls, train_ids, test_feats, test_lbls, test_ids, training_matrix, testing_matrix, unlabeled_matrix, X_train, y_train, X_test, y_test, train_ids_pages, test_ids_pages

def training(features_array, batch_quantity, batch_ids, batch_labels, image_matrix, unlabeled_matrix, ids_pages):
    '''
    Trains the diferent models
    
    @param features_array: array with the features
    @dtype features_array: array
    
    @param batch_quantity: amount of files per batch
    @dtype batch_quantity: int
    
    @param batch_ids: The list of ids to save
    @dtype batch_ids: list
    
    @param batch_labels: The list of labels per file to save
    @dtype batch_labels: list
    
    @param image_matrix: matrix with the image features
    @dtype image_matrix: matrix
    
    @param unlabeled_matrix: matrix with the unlabeled image features
    @dtype unlabeled_matrix: matrix
    
    @param ids_pages: ids of the pages for image processing
    @dtype ids_pages: list
    '''
    train_feats, train_lbls, train_ids, test_feats, test_lbls, test_ids, training_matrix, testing_matrix, \
        unlabeled_matrix, X_train, y_train, X_test, y_test, train_ids_pages, test_ids_pages = preprocess_training_data(features_array, batch_quantity, batch_ids, batch_labels, \
                                                                                                                       paths.DEEP_DATA_PATH, image_matrix, unlabeled_matrix, ids_pages, unlabeled_flag=True) 
                                                                                                                                                        #unlabeled flag for now always True to use pseudolabeling
    models, scores = mo.train_structural_models(train_feats, train_lbls, test_feats, test_lbls, models_path, metadata)
    predictions_prob_train, positive_prob_train = mo.get_models_output(train_feats, models, models_path, metadata)
    predictions_prob_test, positive_prob_test = mo.get_models_output(test_feats, models, models_path, metadata)
    
    predictions_avg_train = None
    predictions_avg_test = None
    if deep:
        models_path_deep = join(models_path,'deep/')
        deep_model = mo.train_deep_model(models_path_deep, training_matrix, y_train, unlabeled_matrix, unlabeled_flag=True)
        predictions_avg_train = mo.get_deep_model_output(training_matrix, deep_model, models_path_deep, train_ids_pages, train_ids)
        predictions_avg_test = mo.get_deep_model_output(testing_matrix, deep_model, models_path_deep, test_ids_pages, test_ids)
        
    train_input_data = [None] * len(train_ids)
    for i,t in enumerate(train_input_data): train_input_data[i] = [] 
    test_input_data = [None] * len(test_ids)
    for i,t in enumerate(test_input_data): test_input_data[i] = []
    for i,f in enumerate(positive_prob_train):
        for j,v in enumerate(positive_prob_train[i]):
            train_input_data[j].append(np.float64(positive_prob_train[i][j]))
        for j,v in enumerate(positive_prob_test[i]):
            test_input_data[j].append(np.float64(positive_prob_test[i][j]))
    if predictions_avg_train and predictions_avg_train:
        for j,v in enumerate(predictions_avg_train):
            train_input_data[j].append(np.float64(predictions_avg_train[j]))
        for j,v in enumerate(predictions_avg_test):
            test_input_data[j].append(np.float64(predictions_avg_test[j]))        
    final_model = mo.train_final_model(train_input_data, train_lbls, models_path, metadata, deep)
    final_score, __ = mo.get_final_model_output(test_input_data, final_model, test_lbls, models_path, metadata, deep)
    logger.info("Final score after training: %s"%final_score)
    
def preprocess_image_data(batch_files, batch_labels, unlabeled_pdfs_path, batch_ids, data_path, load=False, overwrite=False, save=False, train=False):
    '''
    Preprocess or load the image data to prepare for training or prediction
    
    @param batch_files: The list of files to check
    @dtype batch_files: list
    
    @param batch_labels: The list of labels per file to save
    @dtype batch_labels: list
    
    @param unlabeled_pdfs_path: path to where the unlabeled pdfs are located
    @dtype unlabeled_pdfs_path: str
    
    @param batch_ids: The list of ids to save
    @dtype batch_ids: list
    
    @param data_path: path to where the data is stored
    @dtype data_path: str
    
    @param load: whether to load previous preprocessing data or not
    @dtype load: bool
    
    @param overwrite: whether to overwrite previous preprocessing data or not
    @dtype overwrite: bool
    
    @param save: whether to save preprocessing data or not
    @dtype save: bool
    
    @param train: whether it is training or not
    @dtype train: bool
    
    @return image_matrix: matrix with the image features
    @rtype image_matrix: matrix
    
    @return unlabeled_matrix: matrix with the unlabeled image features
    @rtype unlabeled_matrix: matrix
    
    @return all_pages: The amount of pages per file
    @rtype all_pages: list
    
    @return all_unlabeled_pages: The amount of pages per unlabeled file
    @rtype all_unlabeled_pages: list
    
    @return ids_pages: ids of the pages for all files
    @rtype ids_pages: list
    
    @return ids_unlabeled_pages: ids of the pages for all files
    @rtype ids_unlabeled_pages: list
    '''
    if unlabeled_pdfs_path: unlabeled_flag = True
    else: unlabeled_flag = False
    if load:
        images, unlabeled_images, all_pages, all_unlabeled_pages, ids_pages, ids_unlabeled_pages, image_matrix, unlabeled_matrix = da.get_images_from_files(data_path, batch_ids, train)
    else:
        unlabeled_pdf_files = None
        if unlabeled_flag: unlabeled_pdf_files =  da.get_unlabeled_files(unlabeled_pdfs_path)
        
        #First let's check if there are already saved data when wanting to save new data
        if not overwrite and save:
            batch_files, batch_ids, batch_labels, unlabeled_pdf_files = da.check_saved_image_data(data_path, batch_files, batch_ids, batch_labels, unlabeled_pdf_files, train)
        #if there are images to extracted
        image_matrix = []
        unlabeled_matrix = []
        all_pages = []
        all_unlabeled_pages = []
        ids_pages = []
        ids_unlabeled_pages = []
        if batch_files:
            images, unlabeled_images, all_pages, all_unlabeled_pages, ids_pages, ids_unlabeled_pages = da.check_and_extract_images(batch_files, batch_labels, unlabeled_pdf_files, batch_ids, unlabeled_flag, overwrite)
            image_matrix, unlabeled_matrix = df.features_imagenet(images, unlabeled_images, data_path, save, 'all', unlabeled_flag, overwrite)
    return image_matrix, unlabeled_matrix, all_pages, all_unlabeled_pages, ids_pages, ids_unlabeled_pages

def checking_for_saved_data(batch_files, batch_ids, batch_labels, batch_quantity, models_path, features_file, metadata, save_preprocessing=True, train=False):
    '''
    Checks for data that is already saved so that it doesn't extract features from files that are already extracted
    
    @param batch_files: The list of files to check
    @dtype batch_files: list
    
    @param batch_ids: The list of ids to save
    @dtype batch_ids: list
    
    @param batch_labels: The list of labels per file to save
    @dtype batch_labels: list
    
    @param batch_quantity: amount of files per batch
    @dtype batch_quantity: int
    
    @param models_path: path to where the predictive models are located
    @dtype models_path: str
    
    @param features_file: file with the saved features
    @dtype features_file: str
    
    @param metadata: metadata info
    @dtype metadata: list/None
    
    @param save_preprocessing: whether to save preprocessing data or not
    @dtype save_preprocessing: bool
    
    @param train: whether it is training or not
    @dtype train: bool
    
    @param ids_pages: ids of the pages for image processing
    @dtype ids_pages: list
    
    @return remaining_files: The list of files remaining for feature extracting
    @rtype remaining_files: list
    
    @return remaining_ids: The list of ids form remaining files
    @rtype remaining_ids: list
    
    @return remaining_labels: The list of labels of remaining files
    @rtype remaining_labels: list
    
    @return number_of_filtered_files: Number of files that were filtered out
    @rtype number_of_filtered_files: int
    '''
    #Do we need this variable during saving mode? Delete?
    number_of_filtered_files = 0
    remaining_files = batch_files
    remaining_ids = batch_ids
    remaining_labels = batch_labels
    if isfile(features_file):
        logger.info("Checking if there is data already saved. If you want to overwrite data use the -overwrite parameter.")
        debuglogger.info("Checking if there is data already saved. If you want to overwrite data use the -overwrite parameter.")
        __, __, __, __, __, __, remaining_ids, remaining_labels, remaining_indexes = da.load_features(models_path, features_file, metadata, batch_ids, batch_labels, save_preprocessing, train)
        remaining_files = []
        for i in remaining_indexes:
            remaining_files.append(batch_files[i])
        number_of_remaining_files = len(remaining_files)
        number_of_filtered_files = batch_quantity - number_of_remaining_files
        logger.info("Finished checking for saved data. The number of filtered files: %s"%number_of_filtered_files)
        debuglogger.info("Finished checking for saved data. The number of filtered files: %s"%number_of_filtered_files)
    return remaining_files, remaining_ids, remaining_labels, number_of_filtered_files
    
def predicting(features_array, models_path, metadata, image_matrix, ids_pages, batch_ids, deep):
    '''
    Generates the final report
    
    @param features_array: array with the features
    @dtype features_array: array
    
    @param models_path: path to where the predictive models are located
    @dtype models_path: str
    
    @param metadata: metadata info
    @dtype metadata: list/None
    
    @param image_matrix: matrix with the image features
    @dtype image_matrix: matrix
    
    @param ids_pages: ids of the pages for image processing
    @dtype ids_pages: list
    
    @param batch_ids: The list of ids to save
    @dtype batch_ids: list
    
    @param deep: Whether to use deep prediction or not
    @dtype deep: bool
    
    @return final_prediction: Final predictions for the samples
    @rtype final_prediction: array
    '''
    predictions_prob, positive_prob = mo.get_models_output(features_array, None, models_path, metadata)    
    input_data = [None] * len(batch_ids)
    for i,t in enumerate(input_data): input_data[i] = [] 
    for i,f in enumerate(positive_prob):
        for j,v in enumerate(positive_prob[i]):
            input_data[j].append(np.float64(positive_prob[i][j]))
    if deep:
        models_path_deep = join(models_path,'deep/')
        predictions_avg = mo.get_deep_model_output(image_matrix, None, models_path_deep, ids_pages, batch_ids)
        for j,v in enumerate(predictions_avg):
            input_data[j].append(np.float64(predictions_avg[j]))
    __, final_prediction = mo.get_final_model_output(input_data, None, None, models_path, metadata, deep)
    return final_prediction
    
def write_report(results_path, final_prediction, batch_ids, count_pages, t_structure, t_deep, t_pred, courses, number_participants, sample=False):
    '''
    Generates the final report
    
    @param results_path: Path to where the results will be saved
    @dtype results_path: str
    
    @param final_prediction: array with the predictions
    @dtype final_prediction: array
    
    @param batch_ids: The list of ids to save
    @dtype batch_ids: list
    
    @param count_pages: Counter of pages per file
    @dtype count_pages: list
    
    @param t_structure: Time it took to extract the structure per file
    @dtype t_structure: float
    
    @param t_deep: Time it took to extract the deep features per file
    @dtype t_deep: float

    @param t_pred: Time it took to predict the results per file
    @dtype t_pred: float
    
    @param courses: The courses for the files
    @dtype courses: list
    
    @param number_participants: Number of partitipants that had access to each file
    @dtype number_participants: list
    
    @param sample: whether these are results for a sample run
    @dtype sample: bool
    '''
    if courses: courses = list(dict.fromkeys(courses))
    logger.info("Generating final report")
    positive_classified = 0
    positive_classified_list = []
    positive_classified_pages = []
    positive_classified_participants_pages = []
    documents_prob_08 = 0
    documents_prob_list_08 = []
    documents_prob_pages_08 = []
    participants_pages_prob_08 = []
    documents_prob_06 = 0
    documents_prob_list_06 = []
    documents_prob_pages_06 = []
    participants_pages_prob_06 = []
    documents_prob_04 = 0
    documents_prob_list_04 = []
    documents_prob_pages_04 = []
    participants_pages_prob_04 = []
    documents_prob_02 = 0
    documents_prob_list_02 = []
    documents_prob_pages_02 = []
    participants_pages_prob_02 = []
    documents_prob_under_02 = 0
    documents_prob_under_list_02 = []
    documents_prob_under_pages_02 = []
    participants_pages_prob_under_02 = []
    for i,v in enumerate(final_prediction):
        if final_prediction[i] > 0.2:
            documents_prob_02 += 1
            documents_prob_list_02.append(batch_ids[i])
            documents_prob_pages_02.append(count_pages[i])
            participants_pages_prob_02.append(number_participants[i]*count_pages[i])
            if final_prediction[i] > 0.4:
                documents_prob_04 += 1
                documents_prob_list_04.append(batch_ids[i])
                documents_prob_pages_04.append(count_pages[i])
                participants_pages_prob_04.append(number_participants[i]*count_pages[i])
                if final_prediction[i] > 0.5:
                    positive_classified += 1
                    positive_classified_list.append(batch_ids[i])
                    positive_classified_pages.append(count_pages[i])
                    positive_classified_participants_pages.append(number_participants[i]*count_pages[i])
                    if final_prediction[i] > 0.6:
                        documents_prob_06 += 1
                        documents_prob_list_06.append(batch_ids[i])
                        documents_prob_pages_06.append(count_pages[i])
                        participants_pages_prob_06.append(number_participants[i]*count_pages[i])
                        if final_prediction[i] > 0.8:
                            documents_prob_08 += 1
                            documents_prob_list_08.append(batch_ids[i])
                            documents_prob_pages_08.append(count_pages[i])
                            participants_pages_prob_08.append(number_participants[i]*count_pages[i])
        else:
            documents_prob_under_02 += 1
            documents_prob_under_list_02.append(batch_ids[i])
            documents_prob_under_pages_02.append(count_pages[i])
            participants_pages_prob_under_02.append(number_participants[i]*count_pages[i])

    sum_pages_positively_classified = sum(positive_classified_pages)
    sum_pages_over_08 = sum(documents_prob_pages_08)
    sum_pages_over_06 = sum(documents_prob_pages_06)
    sum_pages_over_04 = sum(documents_prob_pages_04)
    sum_pages_over_02 = sum(documents_prob_pages_02)
    sum_pages_under_02 = sum(documents_prob_under_pages_02)
    sum_participants_pages_positively_classified = sum(positive_classified_participants_pages)
    sum_participants_pages_over_08 = sum(participants_pages_prob_08)
    sum_participants_pages_over_06 = sum(participants_pages_prob_06)
    sum_participants_pages_over_04 = sum(participants_pages_prob_04)
    sum_participants_pages_over_02 = sum(participants_pages_prob_02)
    sum_participants_pages_under_02 = sum(participants_pages_prob_under_02)
    #from python 3.6 onwards a standard dict can be used
    from collections import OrderedDict
    report_dict = OrderedDict()
    report_dict['Total documents'] = len(batch_ids)
    report_dict['Positively Classified'] = positive_classified
    report_dict['Probability over 0.8'] = documents_prob_08
    report_dict['Probability over 0.6'] = documents_prob_06
    report_dict['Probability over 0.4'] = documents_prob_04
    report_dict['Probability over 0.2'] = documents_prob_02
    report_dict['Probability under 0.2'] = documents_prob_under_02
    report_dict['Pages Positively Classified'] = sum_pages_positively_classified
    report_dict['Pages Classified over 0.8'] = sum_pages_over_08
    report_dict['Pages Classified over 0.6'] = sum_pages_over_06
    report_dict['Pages Classified over 0.4'] = sum_pages_over_04
    report_dict['Pages Classified over 0.2'] = sum_pages_over_02
    report_dict['Pages Classified under 0.2'] = sum_pages_under_02
    if courses: report_dict['Number of courses'] = len(courses)
    report_dict['Pages x Particpants positively classified'] = sum_participants_pages_positively_classified * sum_pages_positively_classified
    report_dict['Pages x Participants over 0.8'] = sum_participants_pages_over_08 * sum_pages_over_08
    report_dict['Pages x Participants over 0.6'] = sum_participants_pages_over_06 * sum_pages_over_06
    report_dict['Pages x Participants over 0.4'] = sum_participants_pages_over_04 * sum_pages_over_04
    report_dict['Pages x Participants over 0.2'] = sum_participants_pages_over_02 * sum_pages_over_02
    report_dict['Pages x Participants under 0.2'] = sum_participants_pages_under_02 * sum_pages_under_02

    if t_structure != 0: report_dict['Average time preprocessing structure per file'] = t_structure
    if t_deep != 0: report_dict['Average time preprocessing deep features per file'] = t_deep
    if t_pred != 0: report_dict['Average time predicting results per file'] = t_pred
    
    timestr = strftime("%Y%m%d-%H%M%S")
    if not sample: report_name_json = 'report_'+timestr+'.json'
    else: report_name_json = 'report_'+timestr+'_sample.json'
    with open(join(results_path, report_name_json), 'w') as fp:
        json.dump(report_dict, fp)
    if not sample: report_name_csv = 'report_'+timestr+'.csv'
    else: report_name_csv = 'report_'+timestr+'_sample.csv'
    with open(join(results_path,report_name_csv),'w') as fp:
        w = csv.writer(fp)
        w.writerows(report_dict.items())
    logger.info("Final report saved. %s %s"%(report_name_json, report_name_csv))
        
def save_results(results_path, batch_ids, predictions, decisions, sample=False):
    '''
    Saves the prediction results
    
    @param results_path: Path to where the results will be saved
    @dtype results_path: str
    
    @param batch_ids: The list of ids to save
    @dtype batch_ids: list

    @param predictions: array with the predictions
    @dtype predictions: array
    
    @param decisions: array of decisions after classification
    @dtype decisions: array
    
    @param sample: whether these are results for a sample run
    @dtype sample: bool
    '''
    logger.info("Saving prediction results")
    timestr = strftime("%Y%m%d-%H%M%S")
    results_dict = {}
    for i,v in enumerate(final_prediction):
        results_dict[batch_ids[i]+'-probability'] = final_prediction[i]
        results_dict[batch_ids[i]+'-prediction'] = decisions[i]
    if not isdir(results_path): os.makedirs(results_path)
    if not sample: results_name_json = 'results_'+timestr+'.json'
    else: results_name_json = 'results_'+timestr+'_sample.json'
    with open(join(results_path, results_name_json), 'w') as fp:
        json.dump(results_dict, fp)
    if not sample: results_name_csv = 'results_'+timestr+'.csv'
    else: results_name_csv = 'results_'+timestr+'_sample.csv'
    with open(join(results_path,results_name_csv),'w') as fp:
        w = csv.writer(fp)
        w.writerows(results_dict.items())
    logger.info("Final results saved. %s %s"%(results_name_json, results_name_csv))
    
def copy_files_manual_eval(results_path, batch_files, predictions):
    '''
    Saves the prediction results
    
    @param results_path: Path to where the results will be saved
    @dtype results_path: str
    
    @param batch_files: The list of files to copy
    @dtype batch_files: list

    @param predictions: array with the predictions
    @dtype predictions: array
    '''
    manual_path = join(results_path,'positive_manual/')
    logger.info("Copying files classified as positive for manual inspection to %s."%manual_path)
    import random, shutil
    positive_indexes = [i for i,x in enumerate(final_prediction) if x > 0.5]
    if len(positive_indexes) < 100:
        chosen_indexes = positive_indexes
    else: chosen_indexes = random.sample(positive_indexes, 100) 
    if not isdir(manual_path): os.makedirs(manual_path)
    for idx in chosen_indexes:
        shutil.copy2(batch_files[idx], manual_path)
    logger.info("Finished copying the files.")
          
if __name__ == "__main__":
    logger.info("Setting up parameters and environment.")
    args = sys.argv
    params = pa.process_params(args)
    if not params: 
        logger.error("No parameter data. This shouldn't happen.")
        logger.warning("No parameters were loaded. Exiting the application.")
        sys.exit(1)   
    files = params['pdf_files']
    num_files = 0
    all_ids = params['ids']
    load = params['load']
    if files and not load: 
        if type(files) != list and files != None: 
            files = [files]
            all_ids = [all_ids]
            num_files = 1
        else: num_files = len(files)
        files_path = dirname(files[0])
    train = params['train']
    all_labels = params['labels']    
    metadata = params['metadata']
    extra = params['extra']
    overwrite = params['overwrite']
    report = params['report']
    manual = params['manual']
    sample = params['sample']
    cores = params['cores']
    features_file = params['features_file']
    save_preprocessing = params['save_preprocessing']
    deep = params['deep']
    unlabeled_pdfs_path = params['unlabeled_pdfs_path']
    models_path = params['models_path']
    data_path = paths.DATA_PATH
    batch_quantity = 0
    if not train:
        batch_quantity = params['batch']
        decision_threshold = params['threshold']
        results_path = params['results_path'] 
    else: 
        batch_quantity = num_files 
    numeric_features_names = features_names.numeric_features
    if metadata:
        bow_features_names = features_names.bow_text_features + features_names.bow_prop_features +\
        features_names.bow_meta_features
    else:
        bow_features_names = features_names.bow_text_features + features_names.bow_prop_features

    logger.info("Finished configuring parameters and environment.")
    
    if not load:
        logger.info("Using %d core(s)"%cores)
        logger.info("Using batches of %d files"%batch_quantity)
        logger.info("Total number of files to process: %d"%num_files)
        logger.info("Initiating the extraction of features and classification process...")
        if overwrite and save_preprocessing: logger.warning("All saved data will be overwritten!")
        elif save_preprocessing: logger.info("New data will be extracted and concatenated to existing files. Files with extracted data will be preserved.")

    under_batch = 0
    over_batch = batch_quantity
    batch_labels = []
    
    #This are used in the special case where data was saved for the structural features and not for the deep features
    original_files = []
    original_ids = []
    original_labels = []
    number_of_loops = 0
    if batch_quantity > 0: 
        number_of_loops = int(num_files/batch_quantity)
        if (num_files % batch_quantity) > 0: number_of_loops += 1
    
    preprocessing_list = [None] * number_of_loops
    features_array = [None] * number_of_loops
    count_pages = [None] * number_of_loops
    courses = [None] * number_of_loops
    number_participants = [None] * number_of_loops
    #save the value to restore it when checking deep features
    original_overwrite = overwrite
    t_structure = 0
    number_of_processed_files = 0
    if num_files > 0:
        t_structure_0 = time()
        pool_structure = Pool(cores)
        pool_properties = Pool(cores)
        for loop in range(number_of_loops):
            logger.info("Processing batch %d"%loop)
            batch_files = files[under_batch:over_batch]
            batch_ids = all_ids[under_batch:over_batch]
            if train:
                if all_labels:
                    batch_labels = all_labels[under_batch:over_batch]
                else:
                    logger.error("NO LABELS FOUND. Labels are needed for training mode.")
                    debuglogger.error("NO LABELS FOUND. Labels are needed for training mode.")
                    sys.exit(1)
            if not overwrite and save_preprocessing:
                batch_files, batch_ids, batch_labels, number_of_filtered_files = checking_for_saved_data(batch_files, batch_ids, batch_labels, batch_quantity,\
                                                                                                         models_path, features_file, metadata, save_preprocessing, train)
            #Only do the rest if we have files to process
            if batch_files:
                quantity_to_process = len(batch_files)
                logger.info("Number of files in batch to process %s"%quantity_to_process)
                debuglogger.info("Number of files in batch to process %s"%quantity_to_process)
                if quantity_to_process <= 10:
                    logger.info("Files in batch to process: ")
                    logger.info('\n'.join(batch_files))
                    debuglogger.info("Files in batch to process: ")
                    debuglogger.info('\n'.join(batch_files))
                logger.info("Preprocessing starting...")
                logger.info("Checking if data needs to be extracted and extracting it if needed.")
                logger.warning("This could take a long time.")
                preprocessing_list[loop], batch_extra = da.get_preprocessing_data(batch_files, metadata, extra, pool_structure, pool_properties)
                logger.info("Preprocessing finished...")
                logger.info("Extracting features...")
                logger.warning("This could take a long time.")
                features_array[loop], count_pages[loop], courses[loop], number_participants[loop] = da.get_features(batch_ids, preprocessing_list[loop], preprocessing_list[loop][0], models_path, batch_extra, train)
                logger.info("Finished extracting features.")
                logger.info("Normalizing extracted features...")
                features_array[loop][-1] = da.normalize_features(features_array[loop][-1], numeric_features_names, num_files, train) 
                logger.info("Finished normalizing features.")
                if save_preprocessing:
                    logger.info("Saving the extracted features...")
                    e, overwrite = da.save_features(features_array[loop][-1], preprocessing_list[loop], bow_features_names, numeric_features_names, batch_files, batch_ids,\
                                         batch_labels, features_file, count_pages[loop], overwrite)
                    if e:
                        logger.warning("Error while saving the features, check the error log.")
                        debuglogger.error("There was an error while saving the features data: %s.",e)
                number_of_processed_files += quantity_to_process
                logger.info("Total number of files processed: %s from %s"%(number_of_processed_files,num_files))
                debuglogger.info("Total number of files processed: %s from %s"%(number_of_processed_files,num_files))

            under_batch = over_batch
            over_batch = under_batch + batch_quantity
            if num_files < over_batch: over_batch = num_files
        pool_structure.close()
        pool_structure.join()
        pool_properties.close()
        pool_properties.join()
        if features_array[0]:
            features_array = [list(chain.from_iterable(x)) for x in zip(*features_array)]
            features_array = list([np.array(xi) for xi in features_array])
            count_pages = [item for sublist in count_pages for item in sublist]
            courses = [item for sublist in courses for item in sublist]
            number_participants = [int(item) for sublist in number_participants for item in sublist]
        logger.info("Finished processing the structural features of all files...")
        t_structure_1 = time() - t_structure_0
        t_structure = t_structure_1/num_files
        logger.info("Average seconds preprocessing structural features per file: %s"%t_structure)
        #use data in case we still need it to process images
        batch_files = files
        batch_ids = all_ids
        batch_labels = all_labels
        overwrite = original_overwrite
                
    elif load:
        logger.info("Features file specified: %s." %(features_file))
        logger.info("Trying to load saved features.")
        debuglogger.info("Features file specified: %s." %(features_file))
        debuglogger.info("Trying to load saved features.")
        features_array, batch_ids, batch_labels, count_pages, number_participants, course_name, __, __, __ = da.load_features(models_path, features_file, metadata, None, None, False, train)
        batch_quantity = len(batch_ids)
        num_files = batch_quantity
        batch_files = []
    else:
        debuglogger.error("Some parameters are not correct. Either there are no files or load features was not specified.")
        logger.error("Some parameters are not correct. Either there are no files or load features was not specified.")
        sys.exit(1)
                    
    image_matrix = None
    unlabeled_matrix = None
    t_deep = 0
    ids_pages = []
    if deep:
        t_deep_0 = time()
        logger.info("Setting up deep learning data and models.")
        models_path_deep = join(models_path,'deep/')
        image_matrix, unlabeled_matrix, all_pages, all_unlabeled_pages, ids_pages, ids_unlabeled_pages = preprocess_image_data(batch_files, batch_labels,\
                                                                                                                               unlabeled_pdfs_path, batch_ids, paths.DEEP_DATA_PATH, load,\
                                                                                                                               overwrite, save_preprocessing, train)
        t_deep_1 = time() - t_deep_0
        t_deep = t_deep_1/num_files
        logger.info("Average seconds preprocessing deep features per file: %s"%t_deep)

    if train and not save_preprocessing:
        logger.info("Start of training process.")
        t_train_0 = time()
        training(features_array, batch_quantity, batch_ids, batch_labels, image_matrix, unlabeled_matrix, ids_pages)
        t_train_1 = time() - t_train_0
        t_train = t_train_1/num_files
        logger.info("Average seconds training per file: %s"%t_train)
    else: 
        logger.info("Start of prediction process.")
        t_pred_0 = time()
        final_prediction = predicting(features_array, models_path, metadata, image_matrix, ids_pages, batch_ids, deep)
        t_pred_1 = time() - t_pred_0
        t_pred = t_pred_1/num_files
        logger.info("Average seconds predicting per file: %s"%t_pred)
        decisions = []
        for i,v in enumerate(final_prediction):
            if final_prediction[i] >= decision_threshold:
                decisions.append('C')
            else: decisions.append('NC')
        save_results(results_path, batch_ids, final_prediction, decisions, sample)
        if report: write_report(results_path, final_prediction, batch_ids, count_pages, t_structure, t_deep, t_pred, courses, number_participants, sample)
        if manual and batch_files: copy_files_manual_eval(results_path, final_prediction, batch_files)