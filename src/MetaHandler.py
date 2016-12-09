"""
MetaHandler.py

Offers different versions of loading the metadata.
Each functions has an own documentation.
Has some functions for evaluation of trial runs as well.

The main runs an example.

@author: kai
"""
# imports
import os, sys, io

from os.path import join, realpath, dirname, isdir, basename
MOD_PATH = dirname(realpath(__file__))
from doc_globals import*

from datetime import datetime
import random
import numpy as np

import csv
import pandas as pd

# def get_whole_metadata(metafile):
#     """
#     Loads only the classified metadata

#     @param metafile: the file containing the metadata or the full path as string
#     @dtype metafile: string or file

#     @return metadata: the whole metadata
#     @dtype metadata: dict(document_id:list(metadata))
#     """
#     # Make sure the metafiel parameter is of the right type
#     if(isinstance(metafile,str)):
#         try:
#             meta_csv = open(metafile)
#         except IOError:
#             print("The metafile attribute has to be a the filepath or a file opject!")
#             sys.exit(0)
#     else:
#         try:
#             assert(isinstance(metafile,io.IOBase))
#             meta_csv = metafile
#         except AssertionError:
#             print("The metafile attribute has to be a the filepath or a file opject!")
#             sys.exit(0)
#     # Open the csv file
#     reader = csv.DictReader(meta_csv)

#     data_fname2idx = {}

#     # get the fieldnames and parse them in the index dict
#     data_fnames = reader.fieldnames
#     for i in range(0,len(data_fnames)):
#         data_fname2idx[data_fnames[i]] = i

#     # make space for meta_data
#     metadata = {}
#     # read the data row by row
#     for row in reader:
#         # get the row ordered as the fieldnames
#         datarow = [row[name] for name in data_fnames]
#         # make ints to ints
#         datarow[data_fname2idx["filesize"]] = int(datarow[data_fname2idx["filesize"]])
#         # bools to bools
#         datarow[data_fname2idx["author_dr"]] = bool(int(datarow[data_fname2idx["author_dr"]]))
#         datarow[data_fname2idx["author_prof"]] = bool(int(datarow[data_fname2idx["author_prof"]]))
#         datarow[data_fname2idx["is_pdf"]] = bool(int(datarow[data_fname2idx["is_pdf"]]))
#         # and timestamps to timestampts
#         datarow[data_fname2idx["upload_timestamp"]] = datetime.utcfromtimestamp(float(datarow[data_fname2idx["upload_timestamp"]]))
#         # add the datarow
#         metadata[datarow[data_fname2idx["document_id"]]] = datarow

#     # close file if opened by this function
#     if(isinstance(metafile,str)):
#         meta_csv.close()
#     # return the data
#     return metadata

# def get_classified_metadata(metafile, classfile, only_copyright_class=True):
#     """
#     Loads only the metadata of the documents which have been classified and their classification

#     @param metafile: the file containing the metadata or the full path as string
#     @dtype metafile: string or file

#     @param classfile: the file containing the classifications or the full path as string
#     @dtype classfile: string or file

#     @param only_copyright_class: flag deciding if the classifications should only contain the field identifying
#                                 the copyright status or all the other information as well.
#     @dtype only_copyright_class: boolean

#     @return metadata: the metadata as dict
#     @dtype metadata: dict(document_id:list(metadata))

#     @return classifications: the classification as dict
#     @dtype classifications: dict(document_id:classification)
#     """

#     ### Classifications ###
#     # Make sure the classfile parameter is of the right type
#     if(isinstance(classfile,str)):
#         try:
#             class_csv = open(classfile)
#         except IOError:
#             print("The metafile attribute has to be a the filepath or a file opject!")
#             sys.exit(0)
#     else:
#         try:
#             assert(isinstance(classfile,io.IOBase))
#             class_csv = classfile
#         except AssertionError:
#             print("The metafile attribute has to be a the filepath or a file opject!")
#             sys.exit(0)

#     # open the csv reader
#     reader = csv.DictReader(class_csv, delimiter=";")

#     # get all the fielnames and write them to the name->index dict
#     class_fname2idx = {}
#     class_fnames = reader.fieldnames

#     for i in range(0,len(class_fnames)):
#         class_fname2idx[class_fnames[i]] = i

#     # get classifications row by row
#     classifications = {}
#     for row in reader:

#         if(only_copyright_class):
#             classifications[row["document_id"]] = row["published"]=="True"
#         else:
#             row["published"] = row["published"]=="True"
#             classifications[row["document_id"]] = row

#     # close the file if open by this function
#     if(isinstance(classfile,str)):
#         class_csv.close()

#     # check classifiaction if no file exists for the id delete it
#     for c_id in list(classifications):
#         if(not os.path.isfile(join(PDF_PATH, c_id+".pdf"))):
#             del classifications[c_id]

#     ### Metadata ###
#     # Make sure the metafile parameter os of the right type
#     if(isinstance(metafile,str)):
#         try:
#             meta_csv = open(metafile)
#         except IOError:
#             print("The metafile attribute has to be a the filepath or a file opject!")
#             sys.exit(0)
#     else:
#         try:
#             assert(isinstance(metafile,io.IOBase))
#             meta_csv = metafile
#         except AssertionError:
#             print("The metafile attribute has to be a the filepath or a file opject!")
#             sys.exit(0)

#     # open the csv reader
#     reader = csv.DictReader(meta_csv)

#     # get the fieldnames and parse them in the name->index dict
#     data_fname2idx = {}
#     data_fnames = reader.fieldnames
#     for i in range(0,len(data_fnames)):
#         data_fname2idx[data_fnames[i]] = i

#     # make space for meta_data
#     metadata = {}
#     rowdict = {}
#     # read the data row by row
#     for row in reader:
#         # if it has been classified
#         if(row["document_id"] in classifications):
#             rowdict = {}
#             # get the row ordered as the fieldnames
#             datarow = [row[name] for name in data_fnames]
#             # make ints to ints
#             datarow[data_fname2idx["filesize"]] = int(datarow[data_fname2idx["filesize"]])
#             row["filesize"] = datarow[data_fname2idx["filesize"]]
#             # bools to bools
#             datarow[data_fname2idx["author_dr"]] = bool(int(datarow[data_fname2idx["author_dr"]]))
#             row["author_dr"] = datarow[data_fname2idx["author_dr"]]
#             datarow[data_fname2idx["author_prof"]] = bool(int(datarow[data_fname2idx["author_prof"]]))
#             row["author_prof"] = datarow[data_fname2idx["author_prof"]]
#             datarow[data_fname2idx["is_pdf"]] = bool(int(datarow[data_fname2idx["is_pdf"]]))
#             row["is_pdf"] = datarow[data_fname2idx["is_pdf"]]
#             # and timestamps to timestampts
#             datarow[data_fname2idx["upload_timestamp"]] = datetime.utcfromtimestamp(float(datarow[data_fname2idx["upload_timestamp"]]))
#             row["upload_timestamp"] = datarow[data_fname2idx["upload_timestamp"]]
#             metadata[row["document_id"]] = row

#     # close the file if open by this function
#     if(isinstance(meta_csv,str)):
#         meta_csv.close()

#     # return metadata and classification
#     return metadata, classifications

def gen_train_test_split(document_ids, test_size):
    '''
    Splits th provided data into training- and testsplits

    @param  document_ids: The document ids of the classified data
    @type   document_ids: list(string)

    @param  test_size: The fraction of the data supposed to be testing-data
    @type   test_size: float
    '''
    n_data = len(document_ids)
    n_test = int(n_data*test_size)
    doc_copy = document_ids[:]
    random.shuffle(doc_copy)
    train_doc = doc_copy[:n_test]
    test_doc = doc_copy[n_test:]

    return train_doc, test_doc

def trim_classifications(classfile):
    class_csv = open(classfile)
    reader = csv.DictReader(class_csv, delimiter=";")
    fieldnames = reader.fieldnames

    class_meta = get_classified_meta_dataframe()
    class_meta = class_meta["document_id"].tolist()
    meta_dict = {}
    meta_dict = meta_dict.fromkeys(class_meta)
    print(meta_dict)

    with open(join(DATA_PATH, "trimmed_classification.csv"), 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=";")

        writer.writeheader()
        for row in reader:
            doc_id = row["document_id"]
            if(os.path.isfile(join(PDF_PATH, doc_id+".pdf")) and doc_id in meta_dict):
                writer.writerow(row)



    class_csv.close()
    return

def write_classified_metadata(metafile, classfile):
    class_csv = open(classfile)
    reader = csv.DictReader(class_csv, delimiter=";")
    fieldnames = reader.fieldnames
    classifications = {}
    for row in reader:
        classifications[row["document_id"]] = row
    class_csv.close()

    meta_csv = open(metafile)
    reader = csv.DictReader(meta_csv, delimiter=",")
    fieldnames = reader.fieldnames

    with open(join(DATA_PATH, "classified_metadata.csv"), 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=",")
        writer.writeheader()

        for row in reader:
            if(row["document_id"] in classifications):
                writer.writerow(row)

    meta_csv.close()
    return

def get_classified_meta_dataframe(metafile="classified_metadata.csv"):
    meta_data=pd.read_csv(join(DATA_PATH,metafile), header=0, delimiter=',', quoting=0, encoding='utf-8')
    return meta_data

def get_whole_meta_dataframe(metafile="metadata.csv"):
    meta_data=pd.read_csv(join(DATA_PATH,metafile), header=0, delimiter=',', quoting=1, encoding='utf-8')
    return meta_data

def get_trimmed_classifications(class_file="trimmed_classification.csv"):
    class_data=pd.read_csv(join(DATA_PATH,class_file), header=0, delimiter=';', quoting=1, encoding='utf-8')
    return class_data



if __name__ == '__main__':

    trim_classifications(join(DATA_PATH,"classification.csv"))
    # metadata = get_whole_metadata(metafile)
    # print(len(metadata))
    trimmed_classification = get_trimmed_classifications()
    classified_metadata = get_classified_meta_dataframe()
    print(len(classified_metadata))
    print(len(trimmed_classification))

