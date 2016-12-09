# -*- coding: utf-8 -*-

"""
Created on Thu Sep 22 12:16:02 2016

@author: Mats L. Richter
"""

# Feature_Extractor.py

from os.path import basename, splitext

import numpy as np
from multiprocessing import Pool

from doc_globals import*

#INSERT IMPORT features HERE
from features.text_score.text_score import TextScore
from features.bow_pdf.bow_pdf import BoW_Text_Module
from features.page_size_ratio.page_size_ratio import Page_Size_Module
from features.pdf_meta_info.pdf_meta_info import Meta_Info
from features.bow_metadata.bow_metadata import BowMetadata
from features.orientation_detector.orientation_detector import page_orientation_module

from features.negative_bow.negative_bow import Negative_BoW_Text_Module
from features.resolution.resolution import Resolution_Module
from features.meta_pdf_extractor.meta_pdf_extractor import Meta_PDF_Module

#import cProfile
metadata = {}

class FE:
    def __init__(self):
        self.feature_instances = self.init_feature_modules()
        return
        

    def extract_features(self, files, met={}, p=-1):
        global metadata
        metadata = met
        if len(files) == 1:
            res = self.get_data_vector(files[0])
        else:
            with Pool(p) as pool:
                #partial_data_vector = partial(self.get_data_vector, file=files)
                #res = pool.map(partial_data_vector, files, metadata)
                res = pool.map(self.get_data_vector, files)
        return res
    
    #generates the error features
    #
    #@result:   list of features with the error features included
    def generate_error_features(self, features):
        error_feature = 0.0
        for x in features:
            x = np.float64(x)
            if np.isnan(x):
                error_feature = 1.0    
                break
        features.append(error_feature)
        return features
    
    def get_data_vector(self, file):
        feature_data = []
        filepointer = None
        doc_id = splitext(basename(file))[0]   
        try:
            filepointer = open(file,'rb')
        except FileNotFoundError:
            print("doc with id: %s not found!!!" %(doc_id,))
            for fi in self.feature_instances:
                num_feat_vals = len(fi.name)
                feature_data.append([np.nan]*num_feat_vals)
        if(not(filepointer is None)):
            for fi in self.feature_instances:
                num_feat_vals = len(fi.name)
                try:
                    if doc_id in metadata:
                        vals = fi.get_function(filepointer,metadata[doc_id])
                    else:
                        vals = fi.get_function(filepointer,{})

                    if(num_feat_vals==1):
                        feature_data.append(vals)
                    else:
                        feature_data.extend(vals)
                except:
                    print(fi.name)
                    #if error occures the value is nan
                    feature_data.extend([np.nan]*num_feat_vals)
                    continue
    
        feature_data = self.generate_error_features(feature_data)
        feature_data.append(doc_id)
            
        return feature_data
            
    def init_feature_modules(self):
        #initialize module
        modules = []
        
        #with open(join(CONFIG_PATH,'extract_modules.txt')) as em:
        #    modules = em.readlines()
        
        # ADD features HERE
        #for m in modules:
        #    features.append(eval(m.split('\n')[0]))
        # ADD features HERE
        modules.append(TextScore(True))
        modules.append(BoW_Text_Module(True))
        modules.append(Page_Size_Module())
        modules.append(Meta_Info())
        modules.append(page_orientation_module())
        modules.append(BowMetadata("title"))
        modules.append(BowMetadata("author"))
        modules.append(BowMetadata("filename"))
        modules.append(BowMetadata("folder_name"))
        modules.append(BowMetadata("folder_description"))
        modules.append(BowMetadata("description"))
        modules.append(Negative_BoW_Text_Module(True))
        modules.append(Resolution_Module())
        modules.append(Meta_PDF_Module())
    
        #modules.append(OCR_BoW_Module())
    
        return modules