# coding=utf-8
__author__ = 'tkgroot'

import sys, os
from os.path import join, realpath, dirname, isdir, basename, isfile
MOD_PATH = dirname(realpath(__file__))
# from doc_globals import*

from time import time
import re

import nltk
from nltk.corpus import stopwords
nltk.data.path.append(join(MOD_PATH,'nltk_data'))  # setting path to files

import numpy as np
import pandas as pd
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib


class BowMetadata():
    def __init__(self, of_type, punish_factor=0.1, punish_threshold=10, max_features=10, n_estimators=10):
        self.of_type=of_type
        self.name = [of_type]
        self.model = None
        self.vectorizer = None
        self.punish_factor = punish_factor
        self.punish_threshold = punish_threshold
        self.max_features = max_features
        self.n_estimators = n_estimators

    def load_models(self):
        if(not(self.of_type)=="author"):
            vec_file = forest_file = join(MOD_PATH,'vectorizer/'+self.of_type+'.pkl')
            if(not(isfile(forest_file))):
                print("No trained vectorizer for %s. PLease use the training function first." %(self.of_type,))
                sys.exit(1)
            else:
                self.vectorizer = joblib.load(vec_file)

            forest_file = join(MOD_PATH,'forests/'+self.of_type+'.pkl')
            if(not(isfile(forest_file))):
                print("No trained forest for %s. PLease use the training function first." %(self.of_type,))
                sys.exit(1)
            else:
                self.model = joblib.load(forest_file)
        else:
            author_file = join(MOD_PATH,'model_author.csv')
            if(not(isfile(author_file))):
                print("No trained model for author. PLease use the training function first.")
                sys.exit(1)
            else:
                self.model = pd.read_csv(author_file, delimiter=',', header=0, quoting=1)
                self.model = uploader.set_index(['document_id'])
                self.vectorizer = True

    def get_function(self, filepointer, metapointer):
        """
        :param filepointer:
        :param metapointer:
        :return:
        """

        if(not(self.model and self.vectorizer)):
            self.load_models()

        file=basename(filepointer.name)
        file=file[:-4]

        if metapointer is None:
            print("Metapointer is None!")
            return np.nan

        if(self.of_type == 'author'):

            try: # catch label [document_id] which is not in the [index]
                score = self.model['value'].loc[file]
            except:
                score = 0
            if score >= self.punish_threshold:
                return 1
            else:
                return score*self.punish_factor

        else:
            clean_test_data = convert_data(self.metapointer[self.of_type])

            test_data_feature = self.vectorizer.transform(clean_test_data).toarray()
            result_proba = self.model.predict_proba(test_data_feature)

            return result_proba[0][1]

    def train(self, doc_ids, classes):
        if(not(self.of_type == 'author')):

            filepath = join(MOD_PATH,"lib_bow/clean_"+self.of_type+".json")
            data = json.load(open(filepath,"r"))

            clean_data = []
            clf = []

            for d_id, d_cls in zip(doc_ids, classes):
                if(d_id in data):
                    clean_data.append(data[d_id])
                    clf.append(clf)

            print("Counting Words")
            self.vectorizer = CountVectorizer(analyzer='word', encoding="utf-8", max_features=self.max_features)
            self.vectorizer = self.vectorizer.fit(clean_data)

            if(not(isdir(join(MOD_PATH,'voctorizer')))):
                os.makedirs(join(MOD_PATH,'voctorizer'))

            vec_file = join(MOD_PATH,'voctorizer/'+self.of_type+'.pkl')
            joblib.dump(self.vectorizer, vec_file)

            train_data_featues = self.vectorizer.transform(clean_data).toarray()
            print(np.shape(train_data_featues))
            print(train_data_featues[0:2,:])

            # train_data_featues = train_data_featues[0:2,:]
            # clf = clf[0:2]

            print("Training the Forest")
            self.forest = RandomForestClassifier(n_estimators=1, max_features=4,max_depth=10, verbose=0, bootstrap=False, n_jobs=1)
            print("Forest initalized")
            self.forest = self.forest.fit(train_data_featues, clf)
            print("Done")

            if(not(isdir(join(MOD_PATH,'forests')))):
                os.makedirs(join(MOD_PATH,'forests'))

            forest_file = join(MOD_PATH,'forests/'+self.of_type+'.pkl')
            joblib.dump(self.forest, forest_file)

        else:
            train = self.get_train(self.author, self.clf)

            values = list()
            for i,row in enumerate(train.itertuples()):
                values.append(len(train.loc[train['user_id'] == row.user_id]))

            train['value'] = pd.Series(values, index=train.index)

            # Pandas write train to csv file in lib_bow for the bow_author
            train.to_csv(join(MOD_PATH,"model_author.csv"), index=True, quoting=1, encoding='utf-8')

    def get_train(self, data, classifier):
        """
        Training data consists of all metadata from documents who fulfill
        being in data and in classifier
        If the documents doesnt fulfill that requirement it will be NaN
        All NaN´s are dropped from training data and the index is resetet.
        """
        return data.loc[classifier.index]

def generate_clean_training_files(field_name):
    metadata=pd.read_csv(join(DATA_PATH,"classified_metadata.csv"), delimiter=',', quoting=1, encoding='utf-8')
    clf = pd.read_csv(join(DATA_PATH,'trimmed_classification.csv'), delimiter=';', quoting=1, encoding='utf-8')

    metadata=metadata.set_index(['document_id'])
    clf=clf.set_index(['document_id'])

    print("Generating clean training files for "+field_name)

    train = metadata.loc[clf.index]
    clean_train_data = clean_metadata(field_name, train)

    if not os.path.exists(join(MOD_PATH,"lib_bow/")):
        os.makedirs(join(MOD_PATH,'lib_bow/'))

    json_path = join(MOD_PATH,'lib_bow/clean_'+field_name+'.json')

    with open(json_path, 'w') as fp:
        json.dump(clean_train_data, fp, indent=4)

def clean_metadata(index,data):
    clean_data={}
    number_documents=len(data.index)

    for i in range(0,number_documents):
        d_id = data.index[i]
        clean_data[d_id] = convert_data(data.loc[d_id][index])
    return clean_data

def convert_data(data, lang=['german','english']):
    # data is of the format: data[csv-column-name][number of row/document] - ex. data['title'][0]
    try:
        np.isnan(data)
        return""
    except TypeError:
        # removes non-letters&-numbers and keeps Sonderzeichen ä,ö,ü
        # @ToDo: should take care of more special letters from different languages
        text=re.sub(u'[^a-zA-Z0-9\u00fc\u00e4\u00f6]', " ", data)

        all_words=text.lower().split()  # converts all words to lower case letters and splits them
        for language in lang:   # stopwords from given languages are detected and then removed from words
            # @ToDo: integrate Porter Stemming and Lemmatizing for extensive detection of stopwords
            stop_words=set(stopwords.words(language))
            words=[w for w in all_words if not w in stop_words]
        return " ".join(words)
    except:
        print("Data is of a not expected type")
        print(type(data))
        print(data)
        sys.exit(1)

if __name__ == "__main__":

    # sys.path.append("/home/kai/Workspace/deep_doc_class/deep_doc_class/src")
    # from doc_globals import*
    # features = ["title", "filename", "folder_name", "folder_description", "description"]
    # for field_name in features:
    #     generate_clean_training_files(field_name)

    args = sys.argv
    train_file = args[1]
    train = pd.read_csv(train_file, delimiter=',', header=0, quoting=1)
    train.columns = ["class", "document_id"]

    features = []
    features.append(BowMetadata("title"))
    # features.append(BowMetadata("author"))
    features.append(BowMetadata("filename"))
    features.append(BowMetadata("folder_name"))
    features.append(BowMetadata("folder_description"))
    features.append(BowMetadata("description"))

    for f in features:
        f.train(list(train["document_id"]), list(train["class"]))
        print(f.name[0])

