# -*- coding: utf-8 -*-

import os, sys
from os.path import join, realpath, dirname, isdir, basename, isfile

import csv, re, json
import pandas as pd
import numpy as np
import numbers
from multiprocessing import Pool
import subprocess

SRC_DIR = os.path.abspath(join(join(realpath(__file__), os.pardir),os.pardir))
if(not(SRC_DIR in sys.path)):
    sys.path.append(SRC_DIR)
FEATURE_DIR = join(SRC_DIR,"features")
if(not(FEATURE_DIR in sys.path)):
    sys.path.append(FEATURE_DIR)
BOW_DIR = join(SRC_DIR,"bow_classifier")
if(not(BOW_DIR in sys.path)):
    sys.path.append(BOW_DIR)
from bow_classifier import BowClassifier

class Feature_Extractor():

    # list of all possible bow features
    bow_features = [
        #"text",
        "first_100_words",
        "last_100_words",
        "bold_text",
        "producer",
        "creator",
        "filename",
        "folder_name"
    ]

    numeric_features = [
        "count_pages",
        "count_outline_items",
        "count_fonts",
        "count_font_colors",
        "count_font_families",
        "max_font_size",
        "min_font_size",
        "main_font_size",
        "perc_main_font_word",
        "count_images",                # total count
        "total_image_space",          # percentage of total image space
        "dev_image_space_pp",         # std of space per page
        "max_image_space_pp",            # maximum space per page
        "min_image_space_pp",            # minimum space per page
        "biggest_image",              # biggest image
        "samllest_image",             # smallest image
        "count_words",
        "count_bold_words",
        "count_annotations",
        "count_lines",
        "count_textboxes",
        "count_blockstyles",
        "dev_words_pp",
        "dev_lines_pp",
        "dev_textboxes_pp",
        "dev_blockstyles_pp",
        "dev_textbox_space_pp",
        "dev_blockstyle_space_pp",
        "max_words_pp",
        "max_lines_pp",
        "max_textboxes_pp",
        "max_blockstyles_pp",
        "max_textbox_space_pp",
        "max_blockstyle_space_pp",
        "min_words_pp",
        "min_lines_pp",
        "min_textboxes_pp",
        "min_blockstyles_pp",
        "min_textbox_space_pp",
        "min_blockstyle_space_pp",
        "mean_words_per_line",
        "dev_words_per_line",
        "mean_lines_per_blockstyle",
        "dev_lines_per_blockstyle",
        "max_lines_per_blockstyle",
        "modal_right",
        "perc_modal_right",
        "max_right",
        "modal_left",
        "perc_modal_left",
        "max_lefts",
        "modal_textbox_columns_pp",
        "perc_modal_textbox_columns_pp",
        "min_textbox_columns_pp",
        "max_textbox_columns_pp",
        "modal_blockstyle_columns_pp",
        "perc_modal_blockstyle_columns_pp",
        "min_blockstyle_columns_pp",
        "max_blockstyle_columns_pp",
        "total_free_space",
        "dev_free_space_pp",
        "max_free_space_pp",
        "min_free_space_pp",
        'file_size',
        'page_rot',
        'page_size_x',
        'page_size_y'
    ]


    def __init__(self,
        text_dir,
        metadata,
        properties,
        structure):

        self.text_dir = text_dir

        self.metadata_frame = metadata

        self.properties_dict = properties

        self.structure_dict = structure

        self.all_features = self.bow_features + self.numeric_features

        self.bow_classifiers = []
        for bf in Feature_Extractor.bow_features:
            if(bf in self.all_features):
                self.bow_classifiers.append(BowClassifier(bf))

    def extract_features(self, doc_input, feature_file, num_cores=None):
        # Extract Features parallel
        # use the number of cores specified in p
        if(not(num_cores is None) and num_cores>1):
            pool = Pool(p)
            res = pool.map(self.get_data_vector, doc_input)
        # if no cores are specified just use one
        else:
            res = []
            for di in doc_input:
                res.append(self.get_data_vector(di))

        # first fields are the document id and class
        fieldnames = ["document_id", "class"]
        # finally add all the featurenames themselves
        fieldnames += list(self.all_features)

        with open(feature_file,"w") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=',')
            writer.writeheader()
            for row in res:
                if(not(row is None)):
                    writer.writerow(row)

    def get_data_vector(self, t_data):
        doc_id = t_data[0]

        values_dict, bow_strings = self.get_num_vals_and_bow_strings(doc_id)
        if(values_dict is None):
            print("doc with id: %s was not found!!!" %(doc_id,))
            return None

        for bc in self.bow_classifiers:
            #try:
            vals, names = bc.get_function(bow_strings[bc.name],)
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

        values_dict["document_id"] = doc_id
        values_dict["class"] = t_data[1]

        return values_dict

    def get_num_vals_and_bow_strings(self, doc_id):

        # initialize dictionaries for the numeric values and the strings that still have to be evaluated
        values_dict = {}
        bow_strings = {}

        all_features_dict = {}

        with open(join(self.text_dir,doc_id+".txt"), "r") as tf:
            all_features_dict["text"] = tf.read()

        # get csv metadata information
        pd_series = self.metadata_frame[self.metadata_frame['document_id'] == doc_id]
        if(pd_series.empty):
            print("No csv metadata: %s!!!" %(doc_id,))
            all_features_dict["filename"] = "None"
            all_features_dict["folder_name"] = "None"
        else:
            all_features_dict["filename"] = pd_series["filename"]
            all_features_dict["folder_name"] = pd_series["folder_name"]

        # get properties dictionary
        try:
            properties_dict = self.properties_dict[doc_id]
            all_features_dict.update(properties_dict)
        except:
            print("No property data: " + doc_id)
            return None

        # get structure features
        try:
            structure_dict = self.structure_dict[doc_id]
            all_features_dict.update(structure_dict)
        except:
            print("No structure data: " + doc_id)
            return None

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
                elif(type(f_num)==bool):
                    values_dict[key] = int(f_num)
                else:
                    values_dict[f] = None
                    if(type(f_num)!=np.nan and not(f_num is None)):
                        print("Feature %s is not a number!"%(f,))
            else:
                print("Some internal error! %s should not be a possible feature!" (f,))
        return values_dict, bow_strings

    def train_bow_classifier(self, doc_ids, classes, vectorizer, classifier):
        for bc in self.bow_classifiers:
            bc.train(doc_ids, classes, vectorizer, classifier)

if __name__ == "__main__":
    DATA_PATH = "/home/kai/Workspace/deep_doc_class/deep_doc_class/data/"
    PRE_EXTRACTED_DATA_PATH = join(DATA_PATH, "pre_extracted_data")
    class_file = join(DATA_PATH,"training_data.csv")
    with open(class_file, 'r') as df:
        reader = csv.reader(df)
        doc_input = list(reader)

    text_dir = join(DATA_PATH,"xml_text_files")
    metadata=pd.read_csv(join(DATA_PATH,"classified_metadata.csv"), delimiter=',', quoting=1, encoding='utf-8')

    with open(join(PRE_EXTRACTED_DATA_PATH,"pdf_properties_new.json"), "r") as prop_file:
        properties = json.load(prop_file)
    with open(join(PRE_EXTRACTED_DATA_PATH,"xml_text_structure.json"), "r") as struc_file:
        xml_structure = json.load(struc_file)

    fe = Feature_Extractor(
        text_dir=text_dir,
        metadata=metadata,
        properties=properties,
        structure=xml_structure)

    feature_file = join(DATA_PATH, "feature_values", "train_2017_06_11_xml.csv")
    fe.extract_features(doc_input=doc_input,feature_file=feature_file)
