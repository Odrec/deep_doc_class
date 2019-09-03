#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 18:14:42 2017

@author: odrec
"""

from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D, Lambda
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input as rpi
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input as xpi
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as ipi
from keras.preprocessing.image import ImageDataGenerator

import os
import pandas as pd
import logging.config
from time import time
from os.path import isdir, join, isfile
import numpy as np

import paths
logging.config.fileConfig(fname=paths.LOG_FILE, disable_existing_loggers=False)
# Get the logger specified in the file
logger = logging.getLogger("copydocLogger")
debuglogger = logging.getLogger("debugLogger")

def pre_features(MODEL, images_generator, unlabeled_generator, input_tensor, net, lambda_func=None, unlabeled_flag=False):
    x = input_tensor
    if lambda_func: x = Lambda(lambda_func)(x)
    base_model = MODEL(input_tensor=x, weights='imagenet', include_top=False)
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))
    if type(images_generator) != type(None):
        logger.info("Starting prediction of image generator for %s network."%net)
        images = model.predict_generator(images_generator, images_generator.samples)
        logger.info("Finished prediction of image generator for %s network."%net)
        del images_generator
        
    unlabeled = None
    if unlabeled_flag and type(unlabeled_generator) != type(None):
        logger.info("Starting prediction of unlabeled generator for %s network."%net)
        unlabeled = model.predict_generator(unlabeled_generator, unlabeled_generator.samples)
        logger.info("Finished prediction of unlabeled generator for %s network."%net)
        del unlabeled_generator
    return images, unlabeled

def get_generators(images, unlabeled_images, image_size=(224,224), unlabeled_flag=False):
    unlabeled_generator = None
    images_generator = None
    if type(images) != type(None) and not images.empty:
        gen_im = ImageDataGenerator()
        debuglogger.info("Start images generator.")
        if 'label' in images:
            images_generator = gen_im.flow_from_dataframe(images, x_col='image', y_col='label', target_size=image_size, 
                                                  shuffle=False, batch_size=1, steps=images.shape[0], class_mode='other')
        else:
            images_generator = gen_im.flow_from_dataframe(images, x_col='image', target_size=image_size, 
                                                  shuffle=False, batch_size=1, steps=images.shape[0], class_mode=None) 
        debuglogger.info("Finish images generator.")
    if unlabeled_flag and type(unlabeled_images) != type(None) and not unlabeled_images.empty:
        gen_ul = ImageDataGenerator()
        debuglogger.info("Start unlabeled generator.")
        unlabeled_generator = gen_ul.flow_from_dataframe(unlabeled_images, x_col='image', target_size=image_size, 
                                              shuffle=False, batch_size=1, steps=unlabeled_images.shape[0], class_mode=None)
        debuglogger.info("Finished unlabeled generator.")
    width = image_size[0]
    height = image_size[1]
    input_tensor = Input((height, width, 3))
    return images_generator, unlabeled_generator, input_tensor

def features_imagenet(images, unlabeled_images, data_path, save=False, net='all', unlabeled_flag=False, overwrite=False):
    Xim1 = None
    Xim2 = None 
    Xim3 = None
    Xu1 = None
    Xu2 = None
    Xu3 = None
    logger.info("Start generation of deep features.")
    if net == 'res' or net == 'all':     
        t0 = time()
        logger.info("Start generation of res deep features.")
        debuglogger.info("Start generation of res deep features.")
        im_gen, un_gen, input_tensor = get_generators(images, unlabeled_images, (224,224), unlabeled_flag)
        Xim1, Xu1 = pre_features(ResNet50, im_gen, un_gen, input_tensor, 'res', rpi, unlabeled_flag)
        if save: Xim1, Xu1 = save_data_on_files(data_path, Xim1, Xu1, 'res', overwrite, unlabeled_flag)
        t1 = time() - t0
        debuglogger.debug("Generated ResNet features: %s"%t1)
        logger.info("Generated ResNet features: %s"%t1)
    im_gen, un_gen, input_tensor = get_generators(images, unlabeled_images, (299,299), unlabeled_flag)
    if net == 'xce' or net == 'all':
        t0 = time()
        logger.info("Start generation of xce deep features.")
        debuglogger.info("Start generation of xce deep features.")
        Xim2, Xu2 = pre_features(Xception, im_gen, un_gen, input_tensor, 'xce', xpi, unlabeled_flag)
        if save: Xim2, Xu2 = save_data_on_files(data_path, Xim2, Xu2, 'xce', overwrite, unlabeled_flag)
        t1 = time() - t0
        debuglogger.debug("Generated Xception features: %s"%t1)
        logger.info("Generated Xception features: %s"%t1)
    if net == 'inc' or net == 'all':
        t0 = time()
        logger.info("Start generation of inc deep features.")
        debuglogger.info("Start generation of inc deep features.")
        Xim3, Xu3 = pre_features(InceptionV3, im_gen, un_gen, input_tensor, 'inc', ipi, unlabeled_flag)
        if save: Xim3, Xu3 = save_data_on_files(data_path, Xim3, Xu3, 'inc', overwrite, unlabeled_flag)
        t1 = time() - t0
        debuglogger.debug("Generated InceptionV3 features: %s"%t1) 
        logger.info("Generated InceptionV3 features: %s"%t1)
    if net == 'all' and save: 
        images_comp, unlabeled_images_comp = \
        save_labels(data_path, images, unlabeled_images, overwrite, unlabeled_flag)
    return [Xim1, Xim2, Xim3], [Xu1, Xu2, Xu3]

def check_saved_labels(data_path, images):
    '''
    Check saved labels and images.
    
    @param data_path: the path where the data is saved
    @dtype data_path: str
    
    @param images: dataframe with the image paths and labels (labels might be missing for not training process)
    @dtype images: pandas dataframe
    
    @return images: dataframe with the image paths and labels, updated if there were already images on a existing file
    @rtype images: pandas dataframe
    '''
    
    if (type(images) != type(None))  and not images.empty:
        filename = join(data_path,'images.parquet')
        logger.info("Checking for saved file %s."%filename)
        debuglogger.info("Checking for saved file %s."%filename)
        if isfile(filename):
            logger.info("File found! Loading saved data")
            debuglogger.info("File found! Loading saved data")
            images_loaded = pd.read_parquet(filename)
            logger.info("Saved data loaded")
            debuglogger.info("Saved data loaded")
            updated_images = pd.concat([images_loaded, images])
            images = updated_images
            os.remove(filename)
        images.to_parquet(filename)
        logger.info("Finished saving names and labels to file %s."%filename)
        debuglogger.info("Finished saving names and labels to file %s."%filename)
    else:
        debuglogger.debug("No new image data to save!") 
    return images

def save_labels(data_path, images, unlabeled_images, overwrite=False, unlabeled_flag=False):
    '''
    Saves labels and image paths into parquet files.
    
    @param data_path: the path where the data is saved
    @dtype data_path: str
    
    @param images: dataframe with the image paths and labels (labels might be missing for not training process)
    @dtype images: pandas dataframe
    
    @param unlabeled_images: dataframe with the unlabeled image paths
    @dtype unlabeled_images: pandas dataframe
    
    @param overwrite: for overwriting or not saved files
    @dtype overwrite: bool
    
    @param unlabeled_flag: whether we need to save unlabeled images or not
    @dtype unlabeled_flag: bool
    
    @return images: dataframe with the image paths and labels, updated if there were already images on a existing file
    @rtype images: pandas dataframe
    
    @return unlabeled_images: dataframe with the unlabeled image paths, updated if there were already images on a existing file
    @rtype unlabeled_images: pandas dataframe
    '''
    if not isdir(data_path): os.mkdir(data_path)
    if (type(images) != type(None))  and not images.empty:
        filename = join(data_path,'images.parquet')
        logger.info("Saving names and labels to file %s."%filename)
        debuglogger.info("Saving names and labels to file %s."%filename)
        if isfile(filename):
            if overwrite: os.remove(filename)
            else:
                images_loaded = pd.read_parquet(filename)
                updated_images = pd.concat([images_loaded, images], sort=False)
                images = updated_images
                os.remove(filename)
        images.to_parquet(filename)
        logger.info("Finished saving names and labels to file %s."%filename)
        debuglogger.info("Finished saving names and labels to file %s."%filename)
    else:
        debuglogger.debug("No new image data to save!")
        
    if unlabeled_flag:
        if (type(unlabeled_images) != type(None)) and not unlabeled_images.empty:
            filename = join(data_path,'unlabeled_images.parquet')
            logger.info("Saving names to file %s."%filename)
            debuglogger.info("Saving names to file %s."%filename)
            updated_unlabeled_images = None
            if isfile(filename):
                if overwrite: 
                    os.remove(filename)
                    updated_unlabeled_images = unlabeled_images
                else:
                    unlabeled_loaded = pd.read_parquet(filename)
                    updated_unlabeled_images = pd.concat([unlabeled_loaded, unlabeled_images], sort=False)
                    os.remove(filename)
            else: updated_unlabeled_images = unlabeled_images
            updated_unlabeled_images.to_parquet(filename)
            logger.info("Finished saving names to file %s."%filename)
            debuglogger.info("Finished saving names to file %s."%filename)
        else:
            debuglogger.debug("No new unlabeled image data to save!")
    return images, unlabeled_images

def save_data_on_files(data_path, images, unlabeled, net, overwrite=False, unlabeled_flag=False):
    if not isdir(data_path): os.mkdir(data_path)
    filename = join(data_path,"doc_gap_"+net+".parquet")
    logger.info("Saving generator results to file %s."%filename)
    debuglogger.info("Saving generator results to file %s."%filename)
    image_data = images
    image_list = []
    unlabeled_list = []
    if isfile(filename):
        logger.info("File %s found. Loading stored values."%filename)
        debuglogger.info("File %s found. Loading stored values."%filename)
        if overwrite: 
            logger.info("Overwriting file %s."%filename)
            debuglogger.info("Overwriting file %s."%filename)
            os.remove(filename)
        else:
            tmp = pd.read_parquet(filename)
            if 'images' in tmp.columns:
                if image_data.size != 0: 
                    old_images = np.array([np.array(xi) for xi in tmp['images'][0]])
                    logger.info("Concatenating values, size new: %s, size loaded: %s."%(image_data.size, old_images.size))
                    debuglogger.info("Concatenating values, size new: %s, size loaded: %s."%(image_data.size, old_images.size))
                    logger.info("Dimensions new: %s, loaded: %s."%(image_data.shape, old_images.shape))
                    image_data = np.concatenate((old_images, image_data))
                    logger.info("New shape: %s."%image_data.shape[0])
                    image_data = np.array([np.array(xi) for xi in image_data])
                else: image_data = None
            if 'unlabeled' in tmp.columns:
                if unlabeled_flag:
                    old_un = np.array([np.array(xi) for xi in tmp['unlabeled'][0]])
                    if (type(unlabeled) == list and unlabeled):
                        unlabeled = np.array([np.array(xi) for xi in unlabeled])
                    if (type(unlabeled) is np.ndarray and unlabeled.any()): 
                        unlabeled = np.concatenate((old_un, unlabeled))
                    else: unlabeled = None
                else: unlabeled = None

    if not isinstance(image_data, pd.DataFrame) and type(image_data) != type(None): image_list = image_data.tolist()
    if type(unlabeled) != type(None) and type(unlabeled) != list and unlabeled.any(): unlabeled_list = unlabeled.tolist()
    else: unlabeled_list = None
    if type(image_data) != type(None) and type(unlabeled) != type(None) and len(image_data) > 0 and len(unlabeled) > 0:
        df = pd.DataFrame({'images': [image_list], 'unlabeled': [unlabeled_list]})
        df.to_parquet(filename)
        logger.info("Values for images and unlabeled images saved on file %s."%filename)
        debuglogger.debug("Values for images and unlabeled images saved on file %s."%filename)
        logger.info("Finished saving generator results to file %s."%filename)
    elif type(image_data) != type(None) and len(image_data) > 0:
        debuglogger.debug("No unlabeled images data to save.")
        df = pd.DataFrame({'images': [image_list]})
        df.to_parquet(filename)
        logger.info("Values for images saved on file %s."%filename)
        debuglogger.debug("Values for images saved on file %s."%filename)
        logger.info("Finished saving generator results to file %s."%filename)
    elif type(unlabeled) != type(None) and len(unlabeled) > 0:
        debuglogger.debug("No labeled images data to save.")
        df = pd.DataFrame({'unlabeled': [unlabeled_list]})
        df.to_parquet(filename)
        logger.info("Values for unlabeled images saved on file %s."%filename)
        debuglogger.debug("Values for unlabeled images saved on file %s."%filename)
        logger.info("Finished saving generator results to file %s."%filename)
    else:
        logger.info("No new image data to save!")
        debuglogger.info("No new image data to save!")
    del df
    return images, unlabeled

def load_labels(data_path, images, unlabeled_images, overwrite=False, unlabeled_flag=False):
    if not isdir(data_path): os.mkdir(data_path)
    filename = join(data_path,'images.parquet')
    logger.info("Saving names and labels to file %s."%filename)
    new_images = images
    images = new_images
    if isfile(filename):
        if overwrite: os.remove(filename)
        else:
            images_loaded = pd.read_parquet(filename)
            updated_images = pd.concat([images_loaded, images])
            images = updated_images
            os.remove(filename)
    images.to_parquet(filename)

    logger.info("Finished saving names and labels to file %s."%filename)
    if unlabeled_flag:
        filename = join(data_path,'unlabeled_images.parquet')
        logger.info("Saving names to file %s."%filename)
        updated_unlabeled_images = None
        if isfile(filename):
            if overwrite: os.remove(filename)
            else:
                unlabeled_loaded = pd.read_parquet(filename)
                updated_unlabeled_images = pd.concat([unlabeled_loaded, unlabeled_images])
                os.remove(filename)
        else: updated_unlabeled_images = unlabeled_images
        updated_unlabeled_images.to_parquet(filename)
        logger.info("Finished saving names to file %s."%filename)
    return images, unlabeled_images
