# coding=utf-8

import sys, os, pickle
from os.path import join, realpath, abspath, dirname, isfile, isdir

current_path = dirname(abspath(__file__))
sys.path.append(current_path)
import clean_bow_data as cbd

MOD_PATH = dirname(realpath(__file__))
SRC_DIR = abspath(join(join(realpath(__file__), os.pardir),os.pardir))
sys.path.append(SRC_DIR)
sys.path.append(join(SRC_DIR,"features"))

import logging.config

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import joblib

import paths
log_conf_file = join(current_path,'../'+paths.LOG_FILE)
logging.config.fileConfig(fname=log_conf_file, disable_existing_loggers=False)
# Get the logger specified in the file
logger = logging.getLogger("copydocLogger")
debuglogger = logging.getLogger("debugLogger")

class BowClassifier():
    """
    BowClassifier is a container for a trained Bow-Model whose main purpose is to map a input string to a value. 
    The value is the likelyhood for the input string to represent a copyright pdf protected document. 
    The input string can be of different kinds of origins like pdfinfo or some metadata or the content of the pdf. 
    For the classification it uses trained models of a Countvectorizer and a RandomForestClassifier (from the sklearn librabry). 
    The BowClassifier provides means for training, storing and crossvalidating those models as well.
    """
    def __init__(self, classifier="forest"):
        """
        Initializes a BowClassifier.

        @param name: Identifier for what kind of data is mapped in the Analyzer.
        @dtype name: str
        @param classifier: Specification which classifier to use. Either forest, log_reg or word_vector
        @dtype classifier: str
        """
        self.vectorizer = None
        self.classifier = classifier

    def load_vectorizer_model(self, string_type, models_path):
        '''
        Loads the trained vectorizer models for this classifier.

        @param string_type: The vectorizer we need to load the .pk file
        @dtype string_type: str
        '''
        try:
            vectorizers_path = join(models_path, 'vectorizers/')
            vect_file = join(vectorizers_path,'vectorizer_'+string_type+'.pkl')
            debuglogger.debug("Vectorizer to load %s.",vect_file)
            self.vectorizer = joblib.load(vect_file)
            debuglogger.debug("Vectorizer %s succesfully loaded.",vect_file)
        except FileNotFoundError:
            debuglogger.error("Trained vectorizer %s does not exist!" %(vect_file,))
            logger.error("Trained vectorizer %s does not exist!" %(vect_file,))
            sys.exit(1)
            
    def save_vectorizer_model(self, string_type, models_path):
        '''
        Saves the trained vectorizer models.

        @param string_type: The type of vectorizer we need to save.
        @dtype string_type: str
        @param vectorizer: The vectorizer we need to save.
        @dtype string_type: vectorizer
        '''
        vectorizers_path = join(models_path, 'vectorizers/')
        vect_file = join(vectorizers_path, 'vectorizer_'+string_type+'.pkl')
        if isfile(vect_file): logger.warning("Overwriting trained vectorizer %s",vect_file)
        else: 
            if not isdir(vectorizers_path): os.makedirs(vectorizers_path)
            logger.info("Saving trained vectorizer %s",vect_file)
        with open(vect_file, 'wb') as vf:
            pickle.dump(self.vectorizer, vf)        
            
    def get_vectorizer_output(self, doc_ids, data, string_type, models_path, origin_of_data, metadata, train=False):
        '''
        Trains and gets the output from the vectorizerers.

        @param doc_ids: The ids of the documents to be processed.
        @dtype doc_ids: list
        @param data: The BOW data to be processed for each document.
        @dtype data: dict
        @param string_type: The type of vectorizer we need to save.
        @dtype string_type: str
        @param origin_of_data: Origin of data to be processed by the vectorizer.
        @dtype origin_of_data: str
        @param train: Whether is in training mode or not.
        @dtype train: bool
        @return vect_data: The data transformed by the vectorizer.
        @rtype vect_data: array
        '''
        clean_data = []
        # switch string cleaning according to input origin
        if origin_of_data == "meta" and metadata:
            for did in doc_ids: 
                clean_data.append(cbd.preprocess_pdf_metadata_string(data[did][string_type]))
                data[did][string_type] = clean_data[-1]
        elif origin_of_data == "prop":
            for did in doc_ids: 
                clean_data.append(cbd.preprocess_pdf_property_string(data[did][string_type]))
                data[did][string_type] = clean_data[-1]
        elif origin_of_data == "text":
            for did in doc_ids: 
                clean_data.append(cbd.preprocess_pdf_text_string(data[did][string_type]))
                data[did][string_type] = clean_data[-1]            
        if train: 
            debuglogger.debug("Fitting vectorizer for feature %s." %(string_type))
            if origin_of_data == "text": 
                self.vectorizer = TfidfVectorizer(max_features=100, max_df=0.5, min_df=0.01, norm='l2', use_idf=False)
            else:
                self.vectorizer = CountVectorizer(max_features=100, max_df=0.5, min_df=0)
            try:
                vect_data = self.vectorizer.fit_transform(clean_data).toarray()
                self.save_vectorizer_model(string_type, models_path)
            except ValueError:
                debuglogger.warning("Not enough data to train vectorizer for feature %s. Trying to load existing model."\
                                  %(string_type))
                logger.warning("Not enough data to train vectorizer for feature %s. Trying to load existing model."\
                             %(string_type))
                try:
                    self.load_vectorizer_model(string_type, models_path)
                    vect_data = self.vectorizer.transform(clean_data).toarray()
                except:
                    debuglogger.error("Vectorizer for feature %s couldn't be trained or loaded." %(string_type))
                    logger.error("Vectorizer for feature %s couldn't be trained or loaded." %(string_type))
                    sys.exit()
        else:
            self.load_vectorizer_model(string_type, models_path)
            vect_data = self.vectorizer.transform(clean_data).toarray()
        return vect_data

    def load_prediction_model(self, modelpath):
        '''
        Loads the trained forest models for this classifier.

        @param modelpath: The full path to a modelfile (.pkl) file
        @dtype modelpath: str
        '''
        try:
            self.model = joblib.load(modelpath)
        except FileNotFoundError:
            debuglogger.error("Trained model %s does not exist!" %(modelpath,))
            sys.exit(1)
        except:
            debuglogger.error("File %s could not be loaded with sklearn.ensemble.joblib!" %(modelpath,))
            sys.exit(1)
