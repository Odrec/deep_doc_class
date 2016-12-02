# coding=utf-8
__author__ = 'tkgroot'

import sys, os
from os.path import join, realpath, dirname, isdir, basename
MOD_PATH = dirname(realpath(__file__))
if(isdir('/usr/lib/python3.5/lib-dynload')):
    sys.path.append('/usr/lib/python3.5/lib-dynload')
sys.path.append(os.getcwd())
from doc_globals import*

from time import time
import re

import nltk
from nltk.corpus import stopwords
nltk.data.path.append(join(MOD_PATH,'nltk_data'))  # setting path to files

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier


class BowMetadata():
    def __init__(self, of_type=None, punish_factor=None, punish_threshold=None, max_features=None, n_estimators=None):
        if of_type is None: raise ValueError("BowMetadata can't be of type: None")
        else: self.of_type=of_type
        if punish_factor is None: self.punish_factor = 0.1
        if punish_threshold is None: self.punish_threshold = 10
        if max_features is None: max_features=1000
        if n_estimators is None: n_estimators=100

        self.name = [of_type]
        self.vectorizer = CountVectorizer(analyzer='word', max_features=max_features)
        self.forest = RandomForestClassifier(n_estimators=n_estimators)
        ## Initializing BowMetadata if necessary
        # Checks if lib_bow exists, if not it will create it and create clean_train_data
        if not os.path.exists(join(MOD_PATH,"lib_bow/")):
        # if os.path.exists("lib_bow/"):
            os.makedirs(join(MOD_PATH,'lib_bow/'))
        elif not os.path.isfile(join(MOD_PATH,'lib_bow/clean_'+self.of_type+'.txt')) and not self.of_type == 'author':
            t0=time()
            print("Initializing BowMetadata "+self.of_type)
            self.load_metadata_from_csv()

            train = self.get_train(self.metadata,self.clf)
            clean_train_data = self.clean_metadata(self.of_type, train)

            # def classifier_for(self):
            classifier=list()
            for i in range(0, len(train.index)):
                if(i+1)%1000==0: print("Classifier extracted %d of %d\n" % ( i+1, len(train.index)))
                classifier.append(self.clf.loc[train.index[i]].published)
            # for n,index in enumerate(train.index):
            #     for k in range(0,len(train.index)):
            #         if index == self.clf.iloc[k].name:
            #             classifier.append(self.clf.iloc[k].published)
            # print(classifier)

            output_clf=pd.DataFrame(data={'clf':classifier})
            output_clf.to_csv(join(MOD_PATH,'lib_bow/classifier_'+self.of_type+'.csv'), index=False, quoting=1)

            with open(join(MOD_PATH,'lib_bow/clean_'+self.of_type+'.txt'), 'w') as file:
                for row in clean_train_data:
                    file.write(row+'\n')

            print("Done %0.3fs" % (time()-t0))

        elif self.of_type == 'author' and not os.path.isfile(join(MOD_PATH,'lib_bow/model_author.csv')):
            t0=time()
            print("Initializing BowMetadata "+self.of_type)
            self.load_metadata_from_csv()
            self.create_bow_author()
            print("Done %0.3fs" % (time()-t0))
        elif not self.of_type == 'author':
            self.create_bow()

    def load_metadata_from_csv(self):
        # self.metadata=pd.read_csv("tests/metadataTest.csv", header=0, delimiter=',', quoting=1, encoding='utf-8')
        # self.author=pd.read_csv('tests/uploaderTest.csv', header=0, delimiter=",", quoting=1)
        # self.clf=pd.read_csv("tests/classificationTest.csv", header=0, delimiter=';', quoting=3)
        self.metadata=pd.read_csv(join(DATA_PATH,"classified_metadata.csv"), header=0, delimiter=',', quoting=1, encoding='utf-8')
        self.author=pd.read_csv(join(MOD_PATH,'uploader.csv'), header=0, delimiter=",", quoting=1)
        self.clf=pd.read_csv(join(DATA_PATH,"trimmed_classification.csv"), header=0, delimiter=';', quoting=3)

        # self.clf_positive = self.clf.loc[self.clf['published'] == True].reset_index(drop=True)
        # self.clf_negative = self.clf.loc[self.clf['published'] == False]

        # Shift document_id to index
        self.metadata=self.metadata.set_index(['document_id'])
        self.clf=self.clf.set_index(['document_id'])
        self.author=self.author.set_index(['document_id'])

        # Writing Metapointer to csv file - probably not my brightes moment
        # for i, row in enumerate(self.metadata.iterrows()):
        #     self.metadata.iloc[i:(i+1)].to_csv('metapointer/'+self.metadata.iloc[i:(i+1)].index.values[0]+'.csv',
        #                                index=False, header=True, quoting=1, encoding='utf-8')

    def get_train(self, data, classifier):
        """
        Training data consists of all metadata from documents who fulfill
        being in data and in classifier
        If the documents doesnt fulfill that requirement it will be NaN
        All NaN´s are dropped from training data and the index is resetet.
        """
        return data.loc[classifier.index]

    def clean_metadata(self,bow,data):
        clean_data=[]
        number_documents=len(data.index)

        for i in range(0,number_documents):
            if(i+1)%1000==0: print("Review %d of %d\n" % ( i+1, number_documents))
            if self.convert_data(data[bow][i]):
                clean_data.append(self.convert_data(data[bow][i]))
            else:
                clean_data.append('')
        return clean_data

    # data is of the format: data[csv-column-name][number of row/document] - ex. data['title'][0]
    def convert_data(self,data, lang=None):
        if lang is None: lang=['german','english']

        try:    # Necessary to cope with empty descriptions or others. They return NaN if empty
            np.isnan(float(data))
        except:
            # removes non-letters&-numbers and keeps Sonderzeichen ä,ö,ü
            # @ToDo: should take care of more special letters from different languages
            text=re.sub(u'[^a-zA-Z0-9\u00fc\u00e4\u00f6]', " ", data)

            all_words=text.lower().split()  # converts all words to lower case letters and splits them
            for language in lang:   # stopwords from given languages are detected and then removed from words
                # @ToDo: integrate Porter Stemming and Lemmatizing for extensive detection of stopwords
                stop_words=set(stopwords.words(language))
                words=[w for w in all_words if not w in stop_words]
            return " ".join(words)

    def create_bow(self):
        """
        :return:
        """
        with open(join(MOD_PATH,"lib_bow/clean_"+self.of_type+".txt")) as file:
            clean_train_data = [x.strip('\n') for x in file.readlines()]

        read_clf=pd.read_csv(join(MOD_PATH,'lib_bow/classifier_'+self.of_type+'.csv'), header=0, quoting=1, delimiter=',')

        train_data_featues = self.vectorizer.fit_transform(clean_train_data).toarray()
        self.forest = self.forest.fit(train_data_featues, np.ravel(read_clf))

    def create_bow_author(self):
        """
        :return:
        """
        train = self.get_train(self.author, self.clf)

        values = list()
        for i,row in enumerate(train.itertuples()):
            if(i+1)%1000==0: print("Review %d of %d\n" % ( i+1, len(train.index)))
            values.append(len(train.loc[train['user_id'] == row.user_id]))

        train['value'] = pd.Series(values, index=train.index)

        # Pandas write train to csv file in lib_bow for the bow_author
        train.to_csv(join(MOD_PATH,"lib_bow/model_author.csv"), index=True, quoting=1, encoding='utf-8')

    def get_function(self, filepointer, metapointer=None):
        """
        :param filepointer:
        :param metapointer:
        :return:
        """
        if metapointer is None: raise ValueError('Bag of words need a metapointer')
        # print(metapointer)
        # file=re.sub('[^a-zA-Z0-9]','',filepointer)      # get rid of //.*
        file=basename(filepointer.name)     # get rid of //.*
        file=file[:-4]

        if self.of_type == 'author':
            uploader=pd.read_csv(join(MOD_PATH,'lib_bow/model_author.csv'), delimiter=',', header=0, quoting=1)
            uploader = uploader.set_index(['document_id'])

            try: # catch label [document_id] which is not in the [index]
                score = uploader['value'].loc[file]
            except:
                score = 0
            if score >= self.punish_threshold: return 1
            else: return score*self.punish_factor

        # clean_test_data=[]
        # if self.convert_data(metapointer[self.of_type]):
        #     clean_test_data.append(self.convert_data(metapointer[self.of_type]))
        # else:
        #     clean_test_data.append('')

        clean_test_data = self.clean_metadata(self.of_type, metapointer)

        # print(clean_test_data)

        test_data_feature = self.vectorizer.transform(clean_test_data).toarray()
        result = self.forest.predict(test_data_feature)
        result_proba = self.forest.predict_proba(test_data_feature)

        return result_proba[0][1]