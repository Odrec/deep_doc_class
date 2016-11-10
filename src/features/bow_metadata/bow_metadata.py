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


class BowMetadata():
    def __init__(self, of_type=None):
        if of_type is None: raise ValueError("BowMetadata can't be of type: None")
        else: self.of_type=of_type

        self.vectorizer = CountVectorizer(analyzer='word', max_features=1000)
        self.forest = RandomForestClassifier(n_estimators=100)

        ## Initializing BowMetadata if necessary
        # Checks if lib_bow exists, if not it will create it and create clean_train_data
        if not os.path.exists(join(MOD_PATH,"lib_bow/")):
        # if os.path.exists("lib_bow/"):
            os.makedirs(join(MOD_PATH,'lib_bow/'))
        elif not os.path.isfile(join(MOD_PATH,'lib_bow/clean_'+self.of_type+'.txt')):
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

        self.create_bow()

    def load_metadata_from_csv(self):
        # self.metadata=pd.read_csv("tests/metadataTest.csv", header=0, delimiter=',', quoting=1, encoding='utf-8')
        # self.author=pd.read_csv('tests/uploaderTest.csv', header=0, delimiter=",", quoting=1)
        # self.clf=pd.read_csv("tests/classificationTest.csv", header=0, delimiter=';', quoting=3)
        self.metadata=pd.read_csv(join(DATA_PATH,"metadata.csv"), header=0, delimiter=',', quoting=1, encoding='utf-8')
        self.author=pd.read_csv(join(MOD_PATH,'uploader.csv'), header=0, delimiter=",", quoting=1)
        self.clf=pd.read_csv(join(DATA_PATH,"classification.csv"), header=0, delimiter=';', quoting=3)

        # self.clf_positive = self.clf.loc[self.clf['published'] == True].reset_index(drop=True)
        # self.clf_negative = self.clf.loc[self.clf['published'] == False]

        # Shift document_id to index
        self.metadata=self.metadata.set_index(['document_id'])
        self.clf=self.clf.set_index(['document_id'])

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
        with open(join(MOD_PATH,"lib_bow/clean_"+self.of_type+".txt")) as file:
            clean_train_data = [x.strip('\n') for x in file.readlines()]

        read_clf=pd.read_csv(join(MOD_PATH,'lib_bow/classifier_'+self.of_type+'.csv'), header=0, quoting=1, delimiter=',')

        train_data_featues = self.vectorizer.fit_transform(clean_train_data).toarray()
        self.forest = self.forest.fit(train_data_featues, np.ravel(read_clf))

    def get_function(self, filepointer, metapointer=None):
        if metapointer is None: raise ValueError('Bag of words need a metapointer')
        file=basename(filepointer.name)
        file=file[:-4]

        # clean_test_data = self.clean_metadata(self.of_type,metapointer)
        clean_test_data=[]
        if self.convert_data(metapointer[self.of_type]):
            clean_test_data.append(self.convert_data(metapointer[self.of_type]))
        else:
            clean_test_data.append('')
        # print(clean_test_data)

        # classifier=list()
        # for i in range(0,len(self.clf_negative.index)):
        #     classifier.append(self.clf_negative.iloc[i].published)
        # print(classifier)

        test_data_feature = self.vectorizer.transform(clean_test_data).toarray()
        result = self.forest.predict(test_data_feature)
        result_proba = self.forest.predict_proba(test_data_feature)
        # result_tree = self.forest.decision_path(test_data_feature)
        # mean = self.forest.score(test_data_feature, [False,False,False,False])
        # mean = self.forest.score(test_data_feature, [False])
        print(result)
        # print(result_tree)
        print(result_proba)
        # print(mean)
        # print(result[0][0]) #return value
        return result_proba[0][1]

    #old Metadata

    # def bow_author(self):
    #     train=self.get_train(self.author,self.clf)
    #     self.vectorizer.fit_transform(train['user_id']).toarray()
    #     author=self.vectorizer.get_feature_names()
    #     value=np.sum(self.vectorizer.fit_transform(train['user_id']).toarray(),axis=0)
    #
    #     # Pandas creates a lib_bow for the bow_author
    #     output=pd.DataFrame(data={'user_id': author, 'value': value})
    #     output.to_csv( "lib_bow/model_author.csv", index=False, quoting=1, encoding='utf-8')
    #
    # def make_bow(self):
    #     train=self.get_train(self.metadata, self.clf)   # get training data
    #     number_documents=train['document_id'].size      # number of documents in training csv
    #     bows = ['filename', 'title', 'description',
    #             'folder_name', 'folder_description']    # creates bow of given feature, metadata must contain feature
    #     clean_train_data=[]                             # cleaned training data
    #
    #     for bow in bows:
    #         t0=time()
    #         print("create BoW of "+bow)
    #         # go through train data and convert it so it can be used in a bow
    #         for i in range(0,number_documents):   # sets the range to the number of documents
    #             if(i+1)%1000==0: print("Review %d of %d\n" % ( i+1, number_documents))    # notification every 1000 documents
    #             if self.convert_data(train[bow][i]):
    #                 clean_train_data.append(self.convert_data(train[bow][i]))
    #             else:
    #                 clean_train_data.append('')
    #         # print(clean_train_data)
    #
    #         # Create bow
    #         train_data_features=self.vectorizer.fit_transform(clean_train_data).toarray()
    #         bow_vocabulary=self.vectorizer.get_feature_names()
    #
    #         # print(train_data_features)
    #         # print(train_data_features.shape)
    #         # print(train['title'].shape)
    #         # print(len(bow_vocabulary))
    #         # print(bow_vocabulary)
    #
    #         # Sum of the word occurences in bow
    #         word_count=np.sum(train_data_features, axis=0, dtype=np.float64)
    #         word_count=word_count/max(word_count)     # unification of data range[0,1]
    #         # print(word_count)
    #
    #         # Pandas creates a lib_bow for the bow in progress
    #         output=pd.DataFrame( data={'value':word_count, 'word':bow_vocabulary} )
    #         output.to_csv( "lib_bow/model_"+bow+".csv", index=False, quoting=1, encoding='utf-8')
    #         print("finished in: %0.3fs" % (time()-t0))
    #
    # def load_bow_from_csv(self):
    #     bow=pd.read_csv('lib_bow/model_'+self.of_type+'.csv', header=0, delimiter=',', quoting=1, encoding='utf-8')
    #     return bow
    #
    # def get_function(self,filepointer, metapointer=None):
    #     clean_data=[]
    #     file=re.sub('[^a-zA-Z0-9]','',filepointer.name)      # get rid of //.*
    #     file=file[5:-3]                               # cut out 'files' and 'pdf' from pointer str
    #
    #     if self.of_type == 'author':
    #         author=self.author.set_index(['document_id'])        # shift the index to the document_id for easier search
    #         bow=self.load_bow_from_csv()
    #         try:
    #             uploader=author.loc[file].drop_duplicates(keep='first')
    #         except:
    #             return 0
    #         score=bow.loc[bow['user_id'] == uploader['user_id']]['value'].sum()
    #         if score >= self.punish_threshold: return 1
    #         else: return score*self.punish_factor
    #
    #     # load metadata of the file, clean it from artifacts
    #     meta_for_file=self.metadata.loc[self.metadata['document_id'] == file].reset_index(drop=True)
    #     # print(meta_for_file['title'].reset_index(drop=True))
    #
    #     if meta_for_file.empty: # catching empty dataset stating the file doesnt exists in the metadata
    #         return np.nan
    #
    #     clean_data.append(self.convert_data(meta_for_file[self.of_type][0]))
    #     # print(clean_data)
    #     try: # catches empty or nan fields in csv file
    #         self.vectorizer.fit_transform(clean_data).toarray()
    #     except:
    #         return np.nan
    #     data=self.vectorizer.get_feature_names()
    #
    #     # load bow of __of_type
    #     bow=self.load_bow_from_csv()
    #     # print(bow['word'])
    #
    #     # scoring
    #     score=bow.loc[bow['word'].isin(data)].reset_index(drop=True)
    #     size=score.index.size
    #
    #     if size == 0:
    #         return np.nan
    #     else:
    #         return score['value'].sum(axis=0)/size
