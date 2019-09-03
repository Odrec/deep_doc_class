# -*- coding: utf-8 -*-

import os, sys
from os.path import join, realpath

import numpy as np
import numbers
from multiprocessing import Pool
import logging

import features_names

SRC_DIR = os.path.abspath(join(join(realpath(__file__), os.pardir),os.pardir))
if(not(SRC_DIR in sys.path)):
    sys.path.append(SRC_DIR)
FEATURE_DIR = join(SRC_DIR,"features")
if(not(FEATURE_DIR in sys.path)):
    sys.path.append(FEATURE_DIR)
BOW_DIR = join(SRC_DIR,"bow_classifier")
if(not(BOW_DIR in sys.path)):
    sys.path.append(BOW_DIR)
from bow_classifier.bow_classifier import BowClassifier

import paths
#log_conf_file = join(dirname(abspath(__file__)),paths.LOG_FILE)
logging.config.fileConfig(fname=paths.LOG_FILE, disable_existing_loggers=False)
# Get the logger specified in the file
logger = logging.getLogger("copydocLogger")
debuglogger = logging.getLogger("debugLogger")

VECTORIZERS_PATH = '../models/vectorizers/'

class FeatureExtractor():

    def __init__(self,
        doc_ids,
        metadata,
        properties,
        structure,
        train=False):

        self.train = train
        self.doc_ids = doc_ids
        self.metadata_dict = metadata
        self.properties_dict = properties
        self.structure_dict = structure   
        self.bow_text_features = features_names.bow_text_features
        self.bow_prop_features = features_names.bow_prop_features
        if metadata: self.bow_meta_features = features_names.bow_meta_features
        else: self.bow_meta_features = []
        self.bow_features = self.bow_text_features + self.bow_prop_features + self.bow_meta_features
        self.numeric_features = features_names.numeric_features
        self.all_features = self.bow_features + self.numeric_features
        self.feature_values = []
        self.bow_classifiers = []
        
    def get_bow_features(self, models_path):
        self.bow_classifier = BowClassifier()
        for bf in self.bow_features:
            if bf in self.bow_text_features:
                debuglogger.debug("Getting output from vectorizer for text feature %s",bf)
                self.feature_values.append(self.bow_classifier.get_vectorizer_output(self.doc_ids, self.structure_dict, bf, models_path, 'text', self.metadata_dict, self.train))
            elif bf in self.bow_prop_features:
                debuglogger.debug("Getting output from vectorizer for property feature %s",bf)
                self.feature_values.append(self.bow_classifier.get_vectorizer_output(self.doc_ids, \
                                           self.properties_dict, bf, models_path, 'prop', self.metadata_dict, self.train))
            #Have to check if there's metadata at all
            elif bf in self.bow_meta_features and self.metadata_dict:
                debuglogger.debug("Getting output from vectorizer for metadata feature %s",bf)
                self.feature_values.append(self.bow_classifier.get_vectorizer_output(self.doc_ids, \
                                           self.metadata_dict, bf, models_path, 'meta', self.metadata_dict, self.train))
        debuglogger.debug("Number of BOW features: %s", str(len(self.bow_features)))
        debuglogger.debug("BOW features extracted: %s", str(self.bow_features))

    def get_numeric_features(self, extra):
        any_id = list(self.properties_dict.keys())[0]
        tmp_array = np.zeros(shape=(len(self.doc_ids),len(self.numeric_features)), dtype=float)
        count_pages = []
        courses = []
        number_participants = []
        for i,did in enumerate(self.doc_ids):
            if extra:
                courses.append(extra[did]['course_name'])
                number_participants.append(extra[did]['number_participants'])
            for j,nf in enumerate(self.numeric_features):
                if nf in self.properties_dict[any_id]:
                    tmp_array[i][j] = self.properties_dict[did][nf]
                elif nf in self.structure_dict[any_id]:
                    tmp_array[i][j] = self.structure_dict[did][nf] 
                    if nf == 'count_pages':
                        count_pages.append(self.structure_dict[did][nf])
        self.feature_values.append(tmp_array)
        debuglogger.debug("Number of numeric features: %s", str(len(self.numeric_features)))
        debuglogger.debug("Numeric features extracted: %s", str(self.numeric_features))
        return count_pages, courses, number_participants

    def extract_features(self, num_cores=None):
        # Extract Features parallel
        # use the number of cores specified in num_cores
        if num_cores > 1:
            pool = Pool(num_cores)
            res = pool.map(self.get_data_vector,self.doc_ids)
        # if no cores are specified just use one
        else:
            res = []
            for d in self.doc_id: res.append(self.get_data_vector(d))
        return res

    def get_data_vector(self, doc_id):
        values_dict = {}
        values_dict[doc_id], bow_strings = self.get_num_vals_and_bow_strings(doc_id)
        if(values_dict is None):
            debuglogger.error("doc with id: %s was not found!!!" %(doc_id,))
        if self.train: return values_dict
        for bc in self.bow_classifiers:
            vals, names = bc.get_function(bow_strings[bc.name],)
            if type(vals) == list or type(vals) == np.ndarray:
                if len(vals) == len(names):
                    for v,n in zip(vals,names):
                        values_dict[doc_id][n] = np.float64(v)
                else:
                    debuglogger.error("Numbers of names and values don't match!")
                    debuglogger.error(len(vals),len(names))
                    logger.error("Numbers of names and values don't match!")
                    logger.error(len(vals),len(names))
                    sys.exit(1)
            elif type(names) == str and isinstance(vals, numbers.Number):
                values_dict[doc_id][names] = vals
            else:
                debuglogger.error("Unknown return types of get_function! \
                                  names=%s, vals=%s"%(str(type(names)),str(type(vals))))
        return values_dict

    def get_num_vals_and_bow_strings(self, doc_id):

        # initialize dictionaries for the numeric values and the strings that still have to be evaluated
        values_dict = {}
        bow_strings = {}
        all_features_dict = {}        
        all_features_dict["text"] = self.structure_dict[doc_id].pop('text',None)

        if not self.metadata_frame:
            debuglogger.warning("No csv metadata: %s!!!" %(doc_id,))
            all_features_dict["filename"] = "None"
            all_features_dict["folder_name"] = "None"
        else:
            pd_series = self.metadata_frame[doc_id]
            all_features_dict["filename"] = pd_series["filename"]
            all_features_dict["folder_name"] = pd_series["folder_name"]

        # get properties dictionary
        try:
            properties_dict = self.properties_dict[doc_id]
            all_features_dict.update(properties_dict)
        except:
            logger.warning("No property data: " + doc_id)
            return None, None #CHANGE

        # get structure features
        try:
            structure_dict = self.structure_dict[doc_id]
            all_features_dict.update(structure_dict)
        except:
            logger.warning("No structure data: " + doc_id)
            return None, None #CHANGE

        # get the requested information of the dictionary
        for f in self.all_features:
            # check if it is a bow feature or a numeric one
            if(f in self.bow_features):
                f_str = all_features_dict[f]
                # make sure it is a string
                if(type(f_str)!=str and type(f_str)!=np.nan and not(type(f_str) is None)):
                    f_str = str(f_str)
                bow_strings[f] = f_str
            elif(f in FeatureExtractor.numeric_features):
                f_num = all_features_dict[f]
                # make sure it is some kind of number
                if(isinstance(f_num, numbers.Number)):
                    values_dict[f] = f_num
                elif(type(f_num)==bool):
                    values_dict[f] = int(f_num)
                else:
                    values_dict[f] = None
                    if type(f_num) != np.nan and not(f_num is None):
                        logger.warning("Feature %s is not a number!"%(f,))
            else:
                logger.error("Some internal error! %s should not be a possible feature!" (f,))
                debuglogger.error("Some internal error! %s should not be a possible feature!" (f,))
                
        return values_dict, bow_strings

    def train_bow_classifier(self, doc_ids, classes, vectorizer, classifier):
        for bc in self.bow_classifiers:
            bc.train(doc_ids, classes, vectorizer, classifier)
