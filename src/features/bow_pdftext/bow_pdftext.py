import collections
from time import time
import nltk
from nltk.corpus import stopwords
import numpy as np
import re
import pandas
import simplejson as json
from os.path import join, basename
import pickle
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from doc_globals import *
nltk.data.path.append('../data/nltk_data')

__author__ = 'tkgroot'

class bowPDFtext():
    def __init__(self):
        t0 = time()
        print("Initializing BoW PDF Text")
        self.vectorizer = CountVectorizer(analyzer='word', max_features=10000)
        self.forest = RandomForestClassifier(n_estimators=100, warm_start=True)
        self.clf = pandas.read_csv('../data/classification_tests.csv', header=0, quoting=3, delimiter=';', encoding='utf-8')\
            .set_index(['document_id'])
        print("Initializing complete... %0.3fs\n Extracting features from json\n Training the forrest classifier" % (time()-t0))

        clean_file, clean_train_data, classifier = [],[],[]
        failed_docs = []
        for document in range(0, len(self.clf)):
            if(document+1)%1000==0: print("Documents extracted %d of %d\n" % ( document+1, len(self.clf.index)))
            try:
                with open('../data/json_txt_files/'+self.clf.index[document]+'.json') as pdftext:
                    file = json.load(pdftext, object_pairs_hook=collections.OrderedDict)

                for page in file:
                    clean_file.append(self.clean_json(file[page]))

                if ''.join(clean_file):
                    clean_train_data.append(' '.join(clean_file))
                    classifier.append(self.clf.published[document])
                clean_file.clear()
            except:
                failed_docs.append(self.clf.index[document])
                with open('features/bow_pdftext/excluded_documents.txt', 'w') as f:
                    for doc in failed_docs:
                        f.write(doc+"\t"+str(file)+"\n")

        train_data_features = self.vectorizer.fit_transform(clean_train_data).toarray()
        self.forest = self.forest.fit(train_data_features,classifier)
        print("Extracting and Training complete %0.3fs" % (time()-t0))

    def clean_json(self, data, lang=None):
        if lang is None: lang=['german','english']

        text=re.sub(u'[^a-zA-Z0-9\u00fc\u00e4\u00f6]', " ", data)
        all_words=text.lower().split()  # converts all words to lower case letters and splits them
        for language in lang:   # stopwords from given languages are detected and then removed from words
            stop_words=set(stopwords.words(language))
            words=[w for w in all_words if not w in stop_words]
        return " ".join(words)

    def extract(self):
        pass

    def get_function(self, filepointer, metapointer=None):
        file = filepointer.name
        # file=basename(filepointer.name)     # get rid of //.*
        # file=file[:-4]

        clean_file, clean_test_data = [],[]
        try:
            with open('../data/json_txt_files/'+file+'.json') as pdftext:
                file = json.load(pdftext, object_pairs_hook=collections.OrderedDict)

            for page in file:
                clean_file.append(self.clean_json(file[page]))

            if ''.join(clean_file):
                clean_test_data.append(' '.join(clean_file))
            clean_file.clear()
        except:
            return 1

        if not clean_test_data:
            return np.nan
        test_data_feature = self.vectorizer.transform(clean_test_data).toarray()
        result = self.forest.predict(test_data_feature)
        result_proba = self.forest.predict_proba(test_data_feature)
        # print(result, result_proba)
        # return result
        return result_proba[0][1]