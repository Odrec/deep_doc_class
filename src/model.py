#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 17:09:40 2017

@author: odrec
"""

import keras
from os.path import isfile

#@params test_data a list of numpy arrays. Each array is an input
#@params test_labels a numpy array with the target data
def predict(features, model):
    '''
    Makes the predictions for the provided features
    
    @param features: numpy array with the features
    @dtype features: numpy array

    @param model: predicttion model
    @dtype model: keras model
    
    @return prd: prediction
    @rtype prd: numpy array
    '''
    prd = model.predict(features, verbose=0)
    return prd
    
def load_trained_model(args, config_model, metadata):
    '''
    Loads the trained model
    
    @param args: list of arguments passed as parameters
    @dtype args: list

    @param config model: model specified in the cofig file
    @dtype config model: str
    
    @return model: loaded model
    @rtype model: keras model
    
    @return model_path: path to the model
    @rtype model_path: str
    '''
    model = None
    model_path = None
    if isfile(config_model):
        model_path = config_model
    elif '-mod' in args:
        index_model = args.index('-mod') + 1
        model_path = args[index_model]      
    try:
        if model_path is None:
            if metadata:
                model_path = "NN.model"
            else:
                model_path = "NN_noMeta.model"
        model = keras.models.load_model(model_path) 
        return model, model_path
    except:
        return None, model_path