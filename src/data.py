#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 16:57:01 2017

@author: odrec
"""

import csv, json, os
import numpy as np
from os.path import basename, join, splitext, isfile, dirname, realpath

from Feature_Extractor import Feature_Extractor
import features.pdf_properties as fp
import features.pdf_structure as fs

MEANS_PATH = join(join(dirname(realpath(__file__)), os.pardir), "data/features_means")

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
            batch_meta[doc_id] = metadata[doc_id]
        else:
            print("Warning: no metadata found for file with id:",doc_id)
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
    if metadata: batch_meta = get_metadata(batch_files, metadata)
    properties = fp.pre_extract_pdf_properties(batch_files, None, cores)
    text, structure = fs.pre_extract_pdf_structure_values(batch_files, None, None, cores)
    ids = list(properties.keys())
    preprocessing_data = [batch_meta, properties, structure, text]
    return ids, preprocessing_data

def save_preproc_data(doc_ids, preprocessing_list, preprocessing_file):
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
    text = preprocessing_list[3]
    for d in doc_ids:
        preproc_data[d] = {}
        if metadata:
            preproc_data[d]['metadata'] = metadata[d]
        if properties:
            preproc_data[d]['properties'] = properties[d]
        if structure:
            preproc_data[d]['structure'] = structure[d]
        if text:
            text_path = join(preprocessing_path,'text_files')
            text_file = join(text_path, d + '.txt')
            if not isfile(text_file):
                with open(text_file, "w") as text_file:
                    text_file.write(str(text[d]['text']))

    try:
        with open(preprocessing_file, "r") as jsonFile:
            data = json.load(jsonFile)
        data.update(preproc_data)
    except:
        data = preproc_data
    with open(preprocessing_file, "w") as jsonFile:
        json.dump(data, jsonFile)
        
def get_features(doc_ids, preprocessing_list, metadata):
    '''
    Extracts the features
    
    @param doc_ids: The list of doc ids to extract features from
    @dtype doc_ids: list

    @param preprocessing_list: list of dicts with all the preprocessing data
    @dtype preprocessing_list: list
    
    @param metadata: metadata for the doc_ids
    @dtype metadata: str
    '''
    preproc_dict = {}
    for did in doc_ids:
        preproc_dict[did] = {}
        if metadata: preproc_dict[did].update(preprocessing_list[0][did])
        preproc_dict[did].update(preprocessing_list[1][did])
        preproc_dict[did].update(preprocessing_list[2][did])
        preproc_dict[did].update(preprocessing_list[3][did])
    fe = Feature_Extractor(metadata)
    features = fe.extract_bow_features(preproc_dict)
    return features
    
def save_features(features, features_file):
    '''
    Save the features
    
    @param features: dictionary with the features
    @dtype features: dict

    @param features_file: the file on which to store the features
    @dtype features_file: str
    
    @return err: specifies if there was an error reading or opening the file
    @rtype err: bool
    '''
    try:
        with open(features_file, "r") as jsonFile:
            data = json.load(jsonFile)
        data.update(features)
    except:
        data = features
    with open(features_file, "w") as jsonFile:
        json.dump(data, jsonFile)

def replace_nan_mean(features):  
    '''
    Replaces the nan values on the features for the means
    
    @param features: dictionary with the features
    @dtype features: dic
    
    @return max_nor: max value for every feature
    @rtype max_nor: str

    @return min_nor: min value for every feature
    @rtype min_nor: str
    
    @return features: dictionary of features
    @rtype features: dict
    '''
    reader = csv.DictReader(open(join(MEANS_PATH,'output_means.csv'), 'r'))
    means = next(reader)
    min_nor = next(reader)
    max_nor = next(reader)
    for doc_id, feats in features.items():
        for key, val in feats.items():
            if val == np.nan:
                features[doc_id][key] = means[key]
    return max_nor, min_nor
    
def normalize_features(features):
    '''
    Normalizes features
    
    @param features: dictionary with the features
    @dtype features: dic
    '''
    max_nor, min_nor = replace_nan_mean(features)
    for doc_id, feats in features.items():
        for key, val in feats.items():
           if not key == 'error':
               f_range = (np.float64(max_nor[key])-np.float64(min_nor[key]))
               if f_range > 0:
                   features[doc_id][key] = (val-np.float64(min_nor[key]))/f_range
