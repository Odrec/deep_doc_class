# -*- coding: utf-8 -*-

"""
Created on Thu Sep 22 12:16:02 2016

@author: Mats L. Richter
"""

# Feature_Extractor.py
import os, sys
from os.path import join, realpath, dirname, isdir

import csv
import numpy as np
import os, os.path, sys
from multiprocessing import Pool
import MetaHandler
from itertools import repeat
from doc_globals import*

#INSERT IMPORT features HERE
from features.text_score.text_score import TextScore
from features.bow_pdf.bow_pdf import BoW_Text_Module
from features.page_size_ratio.page_size_ratio import Page_Size_Module
from features.scanner_detector.scanner_detector import ScannerDetect
from features.bow_metadata.bow_metadata import BowMetadata
from features.orientation_detector.orientation_detector import page_orientation_module

from features.negative_bow.negative_bow import Negative_BoW_Text_Module
from features.resolution.resolution import Resolution_Module
from features.meta_pdf_extractor.meta_pdf_extractor import Meta_PDF_Module

import cProfile

METADATA = None

def extract_features(data, outfile, p=-1):

    fieldnames = []
    for f in features:
        fieldnames.extend(f.name)

    fieldnames.append("class")
    fieldnames.append("document_id")

    #c,m = get_classes(filenames,classes,metadata)
    
    feat_matrix = list()

    if p == -1:
        pool = Pool()
    else:
        pool = Pool(p)
    
    res = pool.map(get_data_vector, data)

    # # Subtitute for the pool function if time profiling
    # res = []
    # for d in data:
    # 	metapointer = metadata[d]
    # 	res.append(get_data_vector(d))

    # feature_vals = list()
    # file_data = list()
    
    # for item in res:
    #     feature_vals.append(item[1])
    #     file_data.append(item[0]) 
    
    # for f, r in enumerate(feature_vals):
    #     r.append(file_data[f][0])
    #     r.append(file_data[f][1])
    #     feat_matrix.append(r) 

    
    with open(join(FEATURE_VALS_PATH, outfile),"w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=',')
        writer.writeheader()
        for row in res:
            rowdict = {}
            for i, fname in enumerate(fieldnames):
            	rowdict[fname] = row[i]
            writer.writerow(rowdict)
        
    return feat_matrix

#prunes the filenames up to the point, the savefile saved the classifications last time, so the system can proceed
#instead of starting all over from the beginning
def prune(filenames, save_file):
    reader = csv.reader(save_file,delimiter=';')
    for row in reader:
        if row[0] in filenames:
            filenames.remove(row[0])
    return

#generates the error features and replaces the nan values
#
#@result:   list of features with the error features included
def generate_error_features(features):
    
    error_feature = [0.0] * len(features)
    
    for i, x in enumerate(features):
        for j, e in enumerate(x):
            if e == 'nan':
                error_feature[i] = 1.0
                x[j] = 1.0                
    
    features = [x + [error_feature[i]] for i, x in enumerate(features)]
    return features

def get_data_vector(t_data):
	feature_data = []
	filepointer = None
	doc_class = t_data[0]
	doc_id = t_data[1]

	metapointer = METADATA.loc[METADATA['document_id'] == doc_id].reset_index(drop=True)
	try:
		filepointer = open(join(PDF_PATH,doc_id+'.pdf'),'rb')
	except FileNotFoundError:
		print("doc with id: %s not found!!!" %(doc_id,))
		for f in features:
			num_feat_vals = len(f.name)
			feature_data.append([np.nan]*num_feat_vals)
	if(not(filepointer is None)):
		for f in features:
			num_feat_vals = len(f.name)
			try:
				vals = f.get_function(filepointer,metapointer)
				if(num_feat_vals==1):
					feature_data.append(vals)
				else:
					feature_data.extend(vals)
			except:
				print(f.name)

				#if error occures the value is nan
				feature_data.extend([np.nan]*num_feat_vals)
				continue

	feature_data.append(doc_class)
	feature_data.append(doc_id)
	return feature_data

def get_metapointer(path):
    return MetaHandler.get_whole_metadata(path)

#function to train modules if needed. Each module called should have a train function
def train_modules(modules,filenames,classes,metadata):
    #For now modules are pre-trained
    #We want a separate function for this
    for module in modules:
        module.train(filenames,classes,metadata)


if __name__ == "__main__":
	args = sys.argv
	len_args = len(args)

	usage = "python Feature_Extractor.py <datafile.csv> -c <number of cores>"
	data = []

	if(not((len_args==2) or (len_args==4))):
		print(usage)
		sys.exit(1)

	data_file = args[1]
	with open(data_file, 'r') as df:
		reader = csv.reader(df)
		data = list(reader)
		data = data[0:3]

	if(len_args==4 and args[2]=='-c'):
		try:
			p = int(args[3])
		except ValueError:
			print("-c flags needs to be followed by a number")
			print(usage)
			sys.exit(1)
	else:
		p = -1

	#initialize module
	features = list()

	# specify which metafile is to load - default is classified_metadata.csv
	METADATA = MetaHandler.get_classified_meta_dataframe("classified_metadata.csv")

	# ADD features HERE
	features.append(TextScore(True))
	features.append(BoW_Text_Module(True))
	features.append(Page_Size_Module())
	features.append(ScannerDetect())
	features.append(page_orientation_module())
	features.append(BowMetadata("title"))
	features.append(BowMetadata("author"))
	features.append(BowMetadata("filename"))
	features.append(BowMetadata("folder_name"))
	features.append(BowMetadata("folder_description"))
	features.append(BowMetadata("description"))
	features.append(Negative_BoW_Text_Module(True))
	features.append(Resolution_Module())

	#features.append(OCR_BoW_Module())

	print("Extracting Features")
	outfile = "out_test.csv"
	extract_features(data=data,outfile=outfile, p=p)

	# # Getting time spend in all functions called. Doesn't work with multiple threads
	# cProfile.runctx("extract_features(data=doc_ids, features=features, metadata=metadata, p=p)", globals(), locals())