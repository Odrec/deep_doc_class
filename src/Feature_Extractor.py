# -*- coding: utf-8 -*-

import os, sys
from os.path import join, realpath, dirname, isdir, basename, isfile

import csv, re, json
import pandas as pd
import numpy as np
import numbers
from multiprocessing import Pool
from doc_globals import*
from bow_classifier.bow_classifier import BowClassifier

import features.pdf_text
import features.pdf_properties
import features.pdf_structure
import features.pdf_metadata

import time

from PIL import Image as PI
from wand.image import Image

import subprocess

import cProfile, pstats

class Feature_Extractor():

    # list of all possible bow features
    bow_features = ["text",
        "author",
        "producer",
        "creator",
        "title",
        "filename",
        "folder_name",
        "folder_description",
        "description"]

    # list of all possible numeric features
    numeric_features = ['pages',
        'file_size',
        'page_rot',
        'page_size_x',
        'page_size_y',
        'word_count',
        'copyright_symbol',
        'entropy',
        'ratio_text_image',
        'ration_text_pages',
        'ratio_words_box',
        'avg_text_size',
        'ratio_image_pages',
        'avg_image_size']

    def __init__(self,
        pdf_dir,
        text_dir,
        metadata,
        properties,
        structure,
        pdf_properties_features,
        pdf_metadata_features,
        pdf_text_features,
        pdf_structure_features):

        self.pdf_dir = pdf_dir
        self.text_dir = text_dir

        self.metadata_frame = metadata
        if(type(self.metadata_frame)==str and isfile(self.metadata_frame)):
            self.metadata_frame=pd.read_csv(self.metadata_frame, delimiter=',', quoting=1, encoding='utf-8')

        self.properties_dict = properties
        if(type(self.properties_dict)==str and isfile(self.properties_dict)):
            with open(self.properties_dict, "r") as prop_file:
                self.properties_dict = json.load(prop_file)

        self.structure_dict = structure
        if(type(self.structure_dict)==str and isfile(self.structure_dict)):
            with open(self.structure_dict, "r") as prop_file:
                self.structure_dict = json.load(prop_file)

        self.pdf_properties_features = tuple(pdf_properties_features)
        self.pdf_metadata_features = tuple(pdf_metadata_features)
        self.pdf_text_features = tuple(pdf_text_features)
        self.pdf_structure_features = tuple(pdf_structure_features)

        self.all_features = pdf_properties_features + pdf_metadata_features + pdf_text_features + pdf_structure_features

        for f in self.all_features:
            if(not(f in Feature_Extractor.numeric_features or f in Feature_Extractor.bow_features)):
                print("Feature %s is not a valid feature!"%(f,))
                print("Valid features are %s!"%(str(Feature_Extractor.bow_features + Feature_Extractor.numeric_features),))
                sys.exit(1)

        self.bow_classifiers = []
        for bf in Feature_Extractor.bow_features:
            if(bf in self.all_features):
                self.bow_classifiers.append(BowClassifier(bf))

    def extract_features(self, doc_input, feature_file=None, p=-1, profiling=False):

        if(len(doc_input)==1):
            res = [self.get_data_vector(doc_input[0])]

        else:
            if(not(type(doc_input[0])==list)):
                for i in range(len(doc_input)):
                    doc_input[i] = [doc_input[i]]
            # Extract Features parallel
            # use the number of cores specified in p
            if p == -1:
                pool = Pool()
            else:
                pool = Pool(p)

            # if profiling print out how long each function takes
            if(profiling):
                pr = cProfile.Profile()
                pr.enable()
                res = []
                for d in doc_input:
                    res.append(self.get_data_vector(d))
                pr.disable()
                pr.create_stats()
                ps = pstats.Stats(pr).sort_stats('tottime')
                ps.print_stats(0.1)
            # otherwise just get the feature values
            else:
                res = pool.map(self.get_data_vector, doc_input)
            
            # write the dictionary into a csv file
            if(not(feature_file)==None):
                # first field is the document id
                fieldnames = ["document_id"]
                # if the data file contained classifications for training add a column for those
                if(len(doc_input[0])==2):
                    fieldnames += ["class"]
                # finally add all the featurenames themselves
                fieldnames += list(self.all_features)

                with open(feature_file,"w") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=',')
                    writer.writeheader()
                    for row in res:
                        if(not(row is None)):
                            writer.writerow(row)

        res_mat = []
        for val_dict in res:
            res_mat.append(list(val_dict.values()))

        return res_mat

    def generate_error_features(self, features):
        #generates the error features and replaces the nan values
        #
        #@result:   list of features with the error features included
        
        error_feature = [0.0] * len(features)
        
        for i, x in enumerate(features):
            for j, e in enumerate(x):
                if e == 'nan':
                    error_feature[i] = 1.0
                    x[j] = 1.0                
        
        features = [x + [error_feature[i]] for i, x in enumerate(features)]
        return features

    def get_data_vector(self, t_data):

        feature_data = []
        doc_id = t_data[0]

        filepath = join(PDF_PATH,doc_id+'.pdf')

        if(not(isfile(filepath))):
            print(print_bcolors(["WARNING","BOLD"],
                "doc with id: %s was not found!!!" %(doc_id,)))
            return None

        values_dict, bow_strings = self.get_num_vals_and_bow_strings(doc_id)
        if(values_dict is None):
            print(print_bcolors(["WARNING","BOLD"],
                "doc with id: %s is password protected!!!" %(doc_id,)))
            return None

        for bc in self.bow_classifiers:
            #try:
            vals, names = bc.get_function(bow_strings[bc.name],"custom_words_val")
            if(type(vals)==list or type(vals)==np.ndarray):
                if(len(vals)==len(names)):
                    for v,n in zip(vals,names):
                        values_dict[n] = np.float64(v)
                else:
                    print("Numbers of names and values don't match!")
                    print(len(vals),len(names))
                    sys.exit(1)
            elif(type(names)==str and isinstance(vals, numbers.Number)):
                values_dict[names] = vals
            else:
                print("Unknown return types of get_function! names=%s, vals=%s"%(str(type(names)),str(type(vals))))
            #except:
                # print(print_bcolors(["WARNING","BOLD"],
                #     "The BowClassifier for %s failed for id %s!" %(bc.name,doc_id)))
                # values_dict[bc.name] = np.nan

        values_dict["document_id"] = doc_id

        if(len(t_data)==2):
            doc_class = t_data[1]
            values_dict["class"] = doc_class

        return values_dict

    def get_num_vals_and_bow_strings(self, doc_id):

        # initialize dictionaries for the numeric values and the strings that still have to be evaluated
        values_dict = {}
        bow_strings = {}

        all_features_dict = {}

        file_path = join(self.pdf_dir, doc_id+".pdf")
        
        # if some content features are requested
        if(len(self.pdf_text_features)>0):

            text_dict = features.pdf_text.get_pdf_text_features(file_path, text_path=self.text_dir)
            all_features_dict.update(text_dict)

        # if some metadata features are requested
        if(len(self.pdf_metadata_features)>0):
            # try to get csv metadata information for the specific file
            metadata_dict = features.pdf_metadata.load_single_metarow(doc_id, fields=self.pdf_metadata_features, metadata=self.metadata_frame)
            all_features_dict.update(metadata_dict)

        # if some pdf_info features are requested
        if(len(self.pdf_properties_features)>0):
            # get properties dictionary
            if(type(self.properties_dict)==dict and doc_id in self.properties_dict):
                properties_dict = self.properties_dict[doc_id]
            else:
                properties_dict = features.pdf_properties.get_pdf_properties(file_path)
            all_features_dict.update(properties_dict)

        # if some structure features are requested
        if(len(self.pdf_structure_features)>0):
            # get structure dictionary
            if(type(self.structure_dict)==dict and doc_id in self.structure_dict):
                structure_dict = self.structure_dict[doc_id]
            else:
                structure_dict = features.pdf_structure.get_pdf_structure(file_path)
            all_features_dict.update(structure_dict)

        # get the requested information of the dictionary
        for f in self.all_features:
            # check if it is a bow feature or a numeric one
            if(f in Feature_Extractor.bow_features):
                f_str = all_features_dict[f]
                # make sure it is a string
                if(type(f_str)!=str and type(f_str)!=np.nan and not(type(f_str) is None)):
                    f_str = str(f_str)
                bow_strings[f] = f_str
            elif(f in Feature_Extractor.numeric_features):
                f_num = all_features_dict[f]
                # make sure it is some kind of number
                if(isinstance(f_num, numbers.Number)):
                    values_dict[f] = f_num
                else:
                    values_dict[f] = None
                    if(type(f_num)!=np.nan and not(f_num is None)):
                        print("Feature %s is not a number!"%(f,))
            else:
                print("Some internal error! %s should not be a possible feature!" (f,))

        return values_dict, bow_strings
    
    # train bow classifiers

    def train_bow_classifiers(self,filenames,classes):
        #function to train modules if needed. Each module called should have a train function
        #For now modules are pre-trained
        #We want a separate function for this
        for feat in self.bow_classifiers:
            feat.train(filenames,classes)

if __name__ == "__main__":

    args = sys.argv
    len_args = len(args)

    usage = "python Feature_Extractor.py <input_file.csv> <feature_file.csv> [<number of cores>]\n"+ "- the input csv needs to contain document ids in the first and classifications in the second column\n"+\
    "- the output will contain columns for the features the classifications and the document ids in this order\n"+\
    "- the default number of cores is the maximum amount available"
    doc_input = []

    if(not((len_args==3) or (len_args==4))):
        print(usage)
        sys.exit(1)

    input_file = args[1]
    with open(input_file, 'r') as df:
        reader = csv.reader(df)
        doc_input = list(reader)

    feature_file = args[2]

    if(len_args==4):
        try:
            p = int(args[3])
        except ValueError:
            print("<number of cores> needs to be an integer")
            print(usage)
            sys.exit(1)
    else:
        p = -1

    # list of sensible features
    pdf_properties_features = ["producer","creator","pages", "file_size", "page_rot", "page_size_x", "page_size_y"]
    pdf_metadata_features = ["filename","folder_name"]
    pdf_text_features = ["text", "word_count", "copyright_symbol"]
    pdf_structure_features = ['ratio_text_image','ration_text_pages','ratio_words_box','avg_text_size','ratio_image_pages','avg_image_size']

    # # setup loaded dicts
    # pdf_dir = PDF_PATH
    # text_dir = TXT_PATH
    # metadata=pd.read_csv(join(DATA_PATH,"classified_metadata.csv"), delimiter=',', quoting=1, encoding='utf-8')
    # with open(join(PRE_EXTRACTED_DATA_PATH,"pdf_properties.json"), "r") as prop_file:
    #     properties = json.load(prop_file)
    # with open(join(PRE_EXTRACTED_DATA_PATH,"pdf_structure.json"), "r") as struc_file:
    #     structure = json.load(struc_file)

    # # setup files to load
    # pdf_dir = PDF_PATH
    # text_dir = TXT_PATH
    # metadata=join(DATA_PATH,"classified_metadata.csv")
    # properties=join(PRE_EXTRACTED_DATA_PATH,"pdf_properties.json")
    # structure=join(PRE_EXTRACTED_DATA_PATH,"pdf_structure.json")

    # # setup nothing done
    # pdf_dir="/home/kai/Workspace/deep_doc_class/deep_doc_class/data/files_test"
    # text_dir = None
    # metadata=join(DATA_PATH,"classified_metadata.csv")
    # properties = None
    # structure = None

    # setup preextract first
    doc_ids = [d[0] for d in doc_input]

    pdf_dir="/home/kai/Workspace/deep_doc_class/deep_doc_class/data/files_test"
    text_dir = None
    metadata=join(DATA_PATH,"classified_metadata.csv")
    properties = features.pdf_properties.pre_extract_pdf_properties(pdf_dir, doc_ids=None, properties_file=None, num_cores=1)
    structure = features.pdf_structure.pre_extract_pdf_structure_values(pdf_dir, doc_ids=None, structure_file=None, boxinfo_file=None, num_cores=1)

    fe = Feature_Extractor(pdf_dir=pdf_dir,
        text_dir=text_dir,
        metadata=metadata,
        properties=properties,
        structure=structure,
        pdf_properties_features=pdf_properties_features,
        pdf_metadata_features=pdf_metadata_features,
        pdf_text_features=pdf_text_features,
        pdf_structure_features=pdf_structure_features)

    fe.extract_features(doc_input=doc_input,feature_file=feature_file, p=p, profiling=True)