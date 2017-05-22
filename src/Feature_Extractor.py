# -*- coding: utf-8 -*-

import numpy as np
from bow_classifier.bow_classifier import BowClassifier

class Feature_Extractor():
    
    def __init__(self, metadata):
        self.bow_classifiers = []
        if metadata:
            self.bow_classifiers.append(BowClassifier("filename"))
            self.bow_classifiers.append(BowClassifier("folder_name"))
        self.bow_classifiers.append(BowClassifier("creator"))
        self.bow_classifiers.append(BowClassifier("producer"))
        self.bow_classifiers.append(BowClassifier("text"))
            
    def extract_bow_features(self, feature_info):
        # go through the dicts for each document id
        for key,feature_dict in feature_info.items():
            # transform the bow features to values
            for bc in self.bow_classifiers:
                vals, names = bc.get_function(feature_dict[bc.name])
                del feature_dict[bc.name]
                if((not(type(vals)==list or type(vals)==np.ndarray)) and
                   (not(type(names)==list or type(names)==tuple))):
                  names = [names]
                  vals = [vals]
                for n,v in zip(names,vals):
                    feature_info[key][n] = v
            # add error feature
            feature_info[key]["error"] = 0.0    
        
            # make sure every value is transformed to float 64
            for feature_name, feature_value in feature_dict.items():
                try:
                    feature_dict[feature_name] = np.float64(feature_value)
                except:
                    feature_dict[feature_name] = np.nan
                    feature_dict["error"] = 1.0
                    print("Feature %s is not a number but ",str(type(feature_value)),feature_value,feature_name, str(type(feature_name)))
        return feature_info
