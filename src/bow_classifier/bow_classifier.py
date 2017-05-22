# coding=utf-8

import sys, os
from os.path import join, realpath, dirname
MOD_PATH = dirname(realpath(__file__))
SRC_DIR = os.path.abspath(join(join(realpath(__file__), os.pardir),os.pardir))
sys.path.append(SRC_DIR)
sys.path.append(join(SRC_DIR,"features"))

import re

import nltk
from nltk.corpus import stopwords
nltk.data.path.append(join(MOD_PATH,'nltk_data'))  # setting path to files

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib

import math


class BowClassifier():
    """
    BowClassifier is a container for a trained Bow-Model whose main purpose is to map a input string to a value. The value is the likelyhood for the input string to represent a copyright pdf protected document. The input string can be of different kinds of origins like pdfinfo or some metadata or the content of the pdf. For the classification it uses trained models of a Countvectorizer and a RandomForestClassifier (from the sklearn librabry). The BowClassifier provides means for training, storing and crossvalidating those models as well.
    """
    def __init__(self, name, vectorizer="vectorizer", classifier="forest"):
        """
        Initializes a BowClassifier.

        @param name: Identifier for what kind of data is mapped in the Analyzer.
        @dtype name: str
        @param vectorizer: Specification which word vectorizer to use. Either vectorizer or custom.
        @dtype vectorizer: str
        @param classifier: Specification which classifier to use. Either forest, log_reg or word_vector
        @dtype classifier: str
        """
        self.data_origin = None
        self.csvmeta_options = ["title", "folder_name", "description", "folder_description", "filename"]
        self.pdfinfo_options = ["author", "producer", "creator"]

        # check of name is a viable string
        if(name in self.csvmeta_options):
            self.data_origin = "csvmeta"
        elif(name in self.pdfinfo_options):
            self.data_origin = "pdfinfo"
        elif(name=="text"):
            self.data_origin = "pdfcontent"

        # if it was not print usage and exit
        if(self.data_origin is None):
            print("%s is not a valid input argument!!\nUse either one of: %s\nOr one of %s or text!!!"%(self.name,
                str(self.csvmeta_options),str(self.pdfinfo_options)))
            sys.exit(1)

        # set the other fields
        self.name = name

        self.vectorizer = vectorizer
        self.classifier = classifier
        if(self.vectorizer == "vectorizer"):
            self.load_vectorizer_model(join(MOD_PATH,'vectorizer',self.name+'.pkl'))
        elif(self.vectorizer=="custom"):
            self.load_custom_words(join(MOD_PATH, "words", self.name+".txt"))
        else:
            print("%s is not a valid vectorizer. Please use either vectorizer or custom."%(self.vectorizer,))

        if(self.classifier in ["forest","log_reg"]):
            self.load_prediction_model(join(MOD_PATH,vectorizer + "_" + classifier, self.name+'.pkl'))
        elif(self.classifier=="word_vector"):
            pass
        else:
            print("%s is not a valid classifier. Please use either forest, log_reg or word_vector."%(self.classifier,))
            sys.exit(1)

    def load_vectorizer_model(self, modelpath):
        '''
        Loads the trained vectorizer models for this classifier.

        @param modelpath: The full path to a modelfile (.pkl) file
        @dtype modelpath: str
        '''
        try:
            self.vectorizer = joblib.load(modelpath)
        except FileNotFoundError:
            print("File %s does not exist!!!" %(modelpath,))
            sys.exit(1)
        except:
            print("File %s could not be loaded with sklearn.ensemble.joblib!!!" %(modelpath,))
            sys.exit(1)

    def load_prediction_model(self, modelpath):
        '''
        Loads the trained forest models for this classifier.

        @param modelpath: The full path to a modelfile (.pkl) file
        @dtype modelpath: str
        '''
        try:
            self.model = joblib.load(modelpath)
        except FileNotFoundError:
            print("File %s does not exist!!!" %(modelpath,))
            sys.exit(1)
        except:
            print("File %s could not be loaded with sklearn.ensemble.joblib!!!" %(modelpath,))
            sys.exit(1)

    def load_custom_words(self, wordspath):
        try:
            f = open(wordspath, 'r')
            words = f.read()
            f.close()
        except FileNotFoundError:
            print("File %s does not exist!!!" %(wordspath,))
            sys.exit(1)
        except:
            print("Error while reading file %s." %(wordspath,))
            sys.exit(1)
        vocab = re.split("\s",words)
        vocab = re.sub("\s"," ",words)
        vocab = vocab.split()
        self.vectorizer = CountVectorizer(analyzer='word', encoding="utf-8", vocabulary=vocab)

    def get_function(self, input_string, lower_cut=0.25, upper_cut=0.75):
        '''
        Computes the prediction probability for the input string. If the probability is inside the specified lower and upper bound the value is considered to be not specific enough and is changed to 0.5

        @param input_string: The string which is to be classified
        @dtype input_string: str

        @param lower_cut: lower bound for the prediction prbability
        @dtype lower_cut: float

        @param upper_cut: upper bound for the prediction prbability
        @dtype upper_cut: float

        @return f_vals: The probability value or the word occurances vector if the classifier is word_vector
        @rtype  f_vals: float or list(float)

        @return f_names: The name of the feature or a list of words if the classifier is word_vector
        @rtype  f_names: str or list(str)
        '''
        # check if the input is of type string
        if(not(type(input_string)==str or input_string is None)):
            print("Input has to be of type string! It is of type %s" %(str(type(input_string)),))
            sys.exit(1)

        # switch string cleaning according to input origin
        if(self.data_origin == "csvmeta"):
            clean_test_data = preprocess_pdf_metadata_string(input_string)
        elif(self.data_origin == "pdfinfo"):
            clean_test_data = preprocess_pdf_property_string(input_string)
        else:
            clean_test_data = preprocess_pdf_text_string(input_string)

        # get vector for the input
        test_data_feature = self.vectorizer.transform([clean_test_data]).toarray()

        f_names = self.name

        # predict input
        try:
            f_vals = self.model.predict_proba(test_data_feature)[0][1]
            if(f_vals>lower_cut and f_vals<upper_cut):
                f_vals = 0.5
            f_names = self.name
        except AttributeError:
            if(self.classifier=="word_vector"):
                f_vals = test_data_feature[0,:]>0
                f_names = self.vectorizer.get_feature_names()
            else:
                print("Classifier has to be one of %s! It is %s" %
                    (str(["log_reg", "forest","word_vector"]), self.classifier))
                sys.exit(1)

        # return prediction
        return f_vals, f_names

def preprocess_pdf_property_string(text):
    if(text is None):
        return"None"
    else:
        text = text.lower()
        clean_text = "".join(re.findall("[a-z]{2,}",text))
        text = clean_string_regex(text, regex='[^a-z]', sub="")
        return clean_text

def preprocess_pdf_text_string(text):
    text = remove_whitespace(text)
    # words = find_regex(text)
    words = remove_stopwords(text.split())
    text =  " ".join(words)
    return text

def preprocess_pdf_metadata_string(text, lang=['german','english']):
    if(text is None or (type(text) is float and math.isnan(text))):
        return""
    else:
        words = find_regex(text, regex=r'(?u)\b\w\w\w+\b')
        words = remove_stopwords(words)
        return " ".join(words)

def clean_string_regex(txt, regex=';|-|\.|,|\"|[0-9]', sub=""):
    txt = txt.lower()
    txt = re.sub(regex, sub, txt)
    return txt

def remove_whitespace(txt):
    txt = re.sub("\s", " ", txt)
    return txt

def find_regex(txt, regex=r'(?u)\b\w\w\w+\b|©'):
    words = re.findall(regex,txt)
    return words

def remove_stopwords(words):
    languages = ["english", "german", "french"]
    for language in languages:
        stop_words=set(stopwords.words(language))
        words=[w for w in words if not w in stop_words]
    return words
