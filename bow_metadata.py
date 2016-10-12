# coding=utf-8
import nltk, os.path
import pandas as pd
import re, numpy
import numpy as np
from time import time
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
nltk.data.path.append('nltk_data')  # setting path to files

__author__ = 'tkgroot'

class Bow_Metadata():
    def __init__(self, of_type=None, punishment_factor=None):
        if of_type is None: raise ValueError("Bow Metadata can´t be of type: None")
        else: self.of_type=of_type
        if punishment_factor is None: self.punishment_factor=0.5                    # punishment_factor for author only
        else: self.punishment_factor=punishment_factor

        self.type=of_type                                                           # set the type of bow
        self.vectorizer=CountVectorizer(analyzer='word', max_features=1000)         # initialize bow

        # Checks if lib_bow exists, if not it will create it and run make_bow()
        if not os.path.exists("lib_bow/"):
            os.makedirs('lib_bow/')
            self.make_bow()

    def load_data_to_file(self):
        self.metadata=pd.read_csv("metadata.csv", header=0, delimiter=',', quoting=1, encoding='utf-8')
        # self.metadata=pd.read_csv("tests/metadataTest.csv", header=0, delimiter=',', quoting=1, encoding='utf-8')
        self.author=pd.read_csv('uploader.csv', header=0, delimiter=",", quoting=1)
        # self.author=pd.read_csv('tests/uploaderTest.csv', header=0, delimiter=",", quoting=1)
        self.clf=pd.read_csv("classification.csv", header=0, delimiter=';', quoting=3)
        # self.clf=pd.read_csv("tests/classificationTest.csv", header=0, delimiter=';', quoting=3)
        self.clf = self.clf.loc[self.clf['published'] == True]  # consider only positive classification

    # Gets training data
    def get_train(self, data, classifier):
        t0=time()
        print("creating training data...\n")
        """
        Training data consists of all metadata from documents who fulfill
        being in data and in classifier
        If the documents doesnt fulfill that requirement it will be NaN
        All NaN´s are dropped from training data and the index is resetet.
        """
        train=data.loc[data['document_id'].isin(classifier['document_id'])]
        train=train.dropna(how='all').reset_index(drop=True)

        # print(train.dropna(how='all').reset_index())
        print("finished in: %0.3fs" % (time()-t0))
        return train

    # Number of BoWs
    # def create_bow_of(self,names_of_bow=None):
    #     if names_of_bow is None: return [u'filename', u'title', u'description', u'folder_name', u'folder_description']
    #     return names_of_bow

    # data is of the format: data[csv-column-name][number of row/document] - ex. data[u'title'][0]
    def convert_data(self,data, lang=None):
        if lang is None: lang=['german','english']    # can take multiple languages to cross-validate against stopwords

        try:    # Necessary to cope with empty descriptions or others. They return NaN if empty
            np.isnan(float(data))
        except:
            # removes non-letters&-numbers and keeps Sonderzeichen ä,ö,ü
            # @ToDo: should take care of more special letters from different languages
            text=re.sub(u'[^a-zA-Z0-9\u00fc\u00e4\u00f6]', " ", data)

            all_words=text.lower().split()  # converts all words to lower case letters and splits them
            for language in lang:   # stopwords from given languages are detected and then removed from words
                # @ToDo: integrate Porter Stemming and Lemmatizing for extensive detection of stopwords like "messages",
                # "message" as the same word
                stop_words=set(stopwords.words(language))
                words=[w for w in all_words if not w in stop_words]
            return " ".join(words)

    def bow_author(self):
        t0=time()
        print("create BoW of authors...")
        self.load_data_to_file()
        train=self.get_train(self.author,self.clf)
        train.to_csv( "lib_bow/model_author.csv", index=False, quoting=1, encoding='utf-8')
        print("finished in %0.3fs" % (time()-t0))

    # @ToDo: make_bow for negative examples as well
    # @ToDo: might get error when convert_data is empty for some reason. It should be catched
    def make_bow(self):
        self.load_data_to_file()
        train=self.get_train(self.metadata, self.clf)   # get training data
        number_documents=train['document_id'].size      # number of documents in training csv
        bows = ['filename', 'title', 'description',
                'folder_name', 'folder_description']    # creates bow of given feature, metadata must contain feature
        clean_train_data=[]                             # cleaned training data

        for bow in bows:
            t0=time()
            print("create BoW of "+bow)
            # go through train data and convert it so it can be used in a bow
            for i in range(0,number_documents):   # sets the range to the number of documents
                if(i+1)%1000==0: print("Review %d of %d\n" % ( i+1, number_documents))    # notification every 1000 documents
                if self.convert_data(train[bow][i]):
                    clean_train_data.append(self.convert_data(train[bow][i]))
                else:
                    clean_train_data.append('')
            # print(clean_train_data)

            # Create bow
            train_data_features=self.vectorizer.fit_transform(clean_train_data).toarray()
            bow_vocabulary=self.vectorizer.get_feature_names()

            # print(train_data_features)
            # print(train_data_features.shape)
            # print(train['title'].shape)
            # print(len(bow_vocabulary))
            # print(bow_vocabulary)

            # Sum of the word occurences in bow
            word_count=np.sum(train_data_features, axis=0, dtype=np.float64)
            word_count=word_count/max(word_count)     # unification of data range[0,1]
            # print(word_count)

            # Pandas creates a lib_bow for the bow in progress
            output=pd.DataFrame( data={'value':word_count, 'word':bow_vocabulary} )
            output.to_csv( "lib_bow/model_"+bow+".csv", index=False, quoting=1, encoding='utf-8')
            duration=time()-t0
            print("finished in: %0.3fs" % duration)

    # @ToDo: call self.vectorizer.transform(this data) need to change the way vectorizer is used, do prediction instead of score
    def get_function(self,filepointer, metapointer=None):
        clean_data=[]
        self.load_data_to_file()
        file=re.sub('[^a-zA-Z0-9]','',filepointer)      # get rid of //.*
        file=file[5:-3]                               # cut out 'files' and 'pdf' from pointer str

        # load metadata of the file, clean it from artifacts
        meta_for_file=self.metadata.loc[self.metadata['document_id'] == file].reset_index(drop=True)
        # print(meta_for_file['title'].reset_index(drop=True))
        clean_data.append(self.convert_data(meta_for_file[self.of_type][0]))
        print(clean_data)
        self.vectorizer.fit_transform(clean_data).toarray()
        data=self.vectorizer.get_feature_names()

        # load bow of __of_type
        bow=pd.read_csv('lib_bow/model_'+self.of_type+'.csv', header=0, delimiter=',', quoting=1, encoding='utf-8')
        # print(bow['word'])

        # scoring
        score=bow.loc[bow['word'].isin(data)].reset_index(drop=True)
        size=score.index.size

        return score['value'].sum(axis=0)/size

# Testing
# test=Bow_Metadata('title')
# test.make_bow()
# test.bow_author()
# test.get_function("./files/b4825922d723e3e794ddd3036b635420.pdf")