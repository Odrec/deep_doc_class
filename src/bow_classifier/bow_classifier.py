# coding=utf-8
__author__ = 'tkgroot'

import sys, os, shutil
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
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from langdetect import detect_langs
from langdetect import detect
# import treetaggerwrapper as ttwp


class BowClassifier():
    """
    BowClassifier is a container for a trained Bow-Model whose main purpose is to map a input string to a value. The value is the likelyhood for the input string to represent a copyright pdf protected document. The input string can be of different kinds of origins like pdfinfo or some metadata or the content of the pdf. For the classification it uses trained models of a Countvectorizer and a RandomForestClassifier (from the sklearn librabry). The BowClassifier provides means for training, storing and crossvalidating those models as well.
    """
    def __init__(self, name, vec_model=None, prediction_model=None):
        """
        Initializes a BowClassifier. 

        @param name: Identifier for what kind of data is mapped in the Analyzer.
        @dtype name: str
        @param max_features: The maximum amount of words in the Countvectorizer.
        @dtype max_features: int
        @param n_estimators: The amount of trees in the RandomForest.
        @dtype n_estimators: str
        """
        self.data_origin = None
        self.csvmeta_options = ["title", "folder_name", "description", "folder_description", "filename"]
        self.pdfinfo_options = ["author", "producer", "creator"]

        # check of name is a viable string
        if(name in self.csvmeta_options):
            self.data_origin = "csvmeta"
        elif(name in self.pdfinfo_options):
            self.data_origin = "pdfinfo"
        elif(name=="text"):
            self.data_origin = "pdfcontent"

        # if it was not print usage and exit
        if(self.data_origin is None):
            print("%s is not a valid input argument!!\nUse either one of: %s\nOr one of %s or text!!!"(self.name,
                str(self.csvmeta_options),str(self.pdfinfo_options)))
            sys.exit(1)

        # set the other fields
        self.name = name

        self.vectorizer = None
        if(not(vec_model is None)):
            self.vectorizer = load_vectorizer_model(vec_model)
        self.model = None
        if(not(prediction_model is None)):
            self.model = load_prediction_model(prediction_model)

    def load_vectorizer_model(self, modelpath):
        '''
        Loads the trained vectorizer models for this classifier.

        @param modelpath: The full path to a modelfile (.pkl) file
        @dtype modelpath: str
        '''
        try:
            self.vectorizer = joblib.load(modelpath)
        except FileNotFoundError:
            print("File %s does not exist!!!" %(modelpath,))
            sys.exit(1)
        except:
            print("File %s could not be loaded with sklearn.ensemble.joblib!!!" %(modelpath,))
            sys.exit(1)

    def load_prediction_model(self, modelpath):
        '''
        Loads the trained forest models for this classifier.

        @param modelpath: The full path to a modelfile (.pkl) file
        @dtype modelpath: str
        '''
        try:
            self.model = joblib.load(modelpath)
        except FileNotFoundError:
            print("File %s does not exist!!!" %(modelpath,))
            sys.exit(1)
        except:
            print("File %s could not be loaded with sklearn.ensemble.joblib!!!" %(modelpath,))
            sys.exit(1)

    def load_custom_words(self, wordspath):
        try:
            f = open(wordspath, 'r')
            words = f.read()
            f.close()
        except FileNotFoundError:
            print("File %s does not exist!!!" %(wordspath,))
            sys.exit(1)
        except:
            print("Error while reading file %s." %(modelpath,))
            sys.exit(1)
        vocab = re.split("\s",words)
        self.model = vocab.index("")
        vocab = re.sub("\s"," ",words)
        vocab = vocab.split()
        self.vectorizer = CountVectorizer(analyzer='word', encoding="utf-8", vocabulary=vocab)

    def get_function(self, input_string, classifier="log_reg"):
        '''
        Copmputes the prediction probability for the input string.

        @param input_string: The string which is to be classified
        @dtype input_string: str
        '''
        # check if the input is of type string
        if(not(type(input_string)==str or input_string is None)):
            print("Input has to be of type string! It is of type %s" %(str(type(input_string)),))
            sys.exit(1)

        # Load models from the standard path if they do not exist yet.
        if(classifier in ["log_reg", "forest"]):
            if(not(self.vectorizer)):
                self.load_vectorizer_model(join(MOD_PATH,'vectorizer/'+self.name+'.pkl'))
            if(not(self.model)):
                self.load_prediction_model(join(MOD_PATH,classifier+'/'+self.name+'.pkl'))
        elif(classifier in ["custom_words_val", "custom_words_vec"]):
            self.load_custom_words(join(MOD_PATH, "words", self.name+".txt"))
        else:
            print("Classifier has to be one of %s! It is %s" %
                (str(["log_reg", "forest","custom_words_val", "custom_words_vec"]), classifier))
            sys.exit(1)


        # switch string cleaning according to input origin
        if(self.data_origin == "csvmeta"):
            clean_test_data = clean_csv_input(input_string)
        elif(self.data_origin == "pdfinfo"):
            clean_test_data = clean_meta_input(input_string)
        else:
            clean_test_data = clean_pdf_txt_content(input_string)

        # get vector for the input
        test_data_feature = self.vectorizer.transform([clean_test_data]).toarray()

        # predict input
        if(classifier=="log_reg"):
            result_proba = self.model.predict_proba(test_data_feature)[0][1]
        elif(classifier=="forest"):
            result_proba = self.model.predict_proba(test_data_feature)[0][1]
        elif(classifier=="custom_words_val"):
            result_proba = 0.5
            result_proba += (np.sum(test_data_feature[0,0:self.model])>0)*0.5
            result_proba -= (np.sum(test_data_feature[0,self.model:])>0)*0.5
        elif(classifier=="custom_words_vec"):
            result_proba = test_data_feature>0
        else:
            print("Classifier has to be one of %s! It is %s" %
                (str(["log_reg", "forest","custom_words_val", "custom_words_vec"]), classifier))
            sys.exit(1)
        
        # return prediction
        return result_proba

    def train(self, doc_ids, classes, classifier="log_reg"):
        '''
        Trains the vectorizer and forest.

        @param doc_ids: The list of document ids which should be used for training
        @dtype doc_ids: list(str)
        @param classes: The classifications for the documents ids in the same order as the doc_ids
        @dtype classes: list(bool)
        '''
        # check input arguments
        if(not(type(doc_ids)==list)):
            print("Input argument <input_string> has to be of type list! It is of type %s" %(str(type(doc_ids)),))
            sys.exit(1)
        if(not(type(classes)==list or type(classes)==np.ndarray)):
            print("Input argument <input_string> has to be of type list or np.ndarray! It is of type %s" %(str(type(doc_ids)),))
            sys.exit(1)

        # get cleaned data with classification
        # switch data origin
        if(self.data_origin == "csvmeta"):
            clean_data, clf = load_data_csvmeta(doc_ids,classes, self.name)
        elif(self.data_origin == "pdfinfo"):
            clean_data, clf = load_data_pdfinfo(doc_ids,classes,self.name)
        else:
            clean_data, clf = load_data_pdfcontent(doc_ids,classes,num_pages=1)

        clf = np.array(clf)

        # # Analyse the distribution of words to the different classification
        # self.analyze_word_distribution(clean_data,clf)
        # pause()
        # return

        if(classifier=="log_reg"):
            self.vectorizer = CountVectorizer(analyzer='word',
                token_pattern=r'(?u)\b\w\w\w+\b|©',
                max_features=10000,
                encoding="utf-8",
                max_df=0.9,
                min_df=0.003)
            self.vectorizer = self.vectorizer.fit(clean_data)
            train_data_featues = self.vectorizer.transform(clean_data).toarray()

            self.model = LogisticRegression(penalty='l2', C=1, fit_intercept=True, intercept_scaling=1000)
            self.model.fit(train_data_featues, clf)

            # Print important words
            coeffs = self.model.coef_.reshape(len(train_data_featues[0]),)
            vec_words = self.vectorizer.get_feature_names()
            lg_coefs_zipped = zip(vec_words,coeffs)
            sorted_coefs = sorted(lg_coefs_zipped,key=lambda idx: idx[1],reverse=True)
            for i in range(min(100,len(sorted_coefs))):
                print("%20s\t%3f"%(sorted_coefs[i][0],sorted_coefs[i][1]))
            for i in range(len(sorted_coefs)-min(100,len(sorted_coefs)),len(sorted_coefs)):
                print("%20s\t%3f"%(sorted_coefs[i][0],sorted_coefs[i][1]))
            sys.exit()

        elif(classifier=="forest"):
            self.vectorizer = CountVectorizer(analyzer='word',
                token_pattern=r'(?u)\b\w\w\w+\b|©',
                max_features=10000,
                encoding="utf-8",
                max_df=0.9,
                min_df=0.003)
            self.vectorizer = self.vectorizer.fit(clean_data)
            train_data_featues = self.vectorizer.transform(clean_data).toarray()

            self.model = RandomForestClassifier(n_estimators=100)
            self.model = self.model.fit(train_data_featues, clf)

        elif(classifier=="custom_words_val"):
            wordspath = join(MOD_PATH, "words", self.name+".txt")
            self.load_custom_words(wordspath)
            train_data_featues = self.vectorizer.transform(clean_data).toarray()


        elif(classifier=="custom_words_vec"):
            wordspath = join(MOD_PATH, "words", self.name+".txt")
            self.load_custom_words(wordspath)
            train_data_featues = self.vectorizer.transform(clean_data).toarray()

        else:
            print("<classifier> has to be one of [log_reg, forest, custom]. It is %s!!!"%(classifier,))
            sys.exit(1)

        if(not(isdir(join(MOD_PATH,'vectorizer')))):
            os.makedirs(join(MOD_PATH,'vectorizer'))

        vec_file = join(MOD_PATH,'vectorizer/'+self.name+'.pkl')
        joblib.dump(self.vectorizer, vec_file)

        if(not(isdir(join(MOD_PATH,classifier)))):
            os.makedirs(join(MOD_PATH,classifier))

        model_file = join(MOD_PATH,classifier+'/'+self.name+'.pkl')
        joblib.dump(self.model, model_file)

    def analyze_word_distribution(self,clean_data,clf):
            pos_vectorizer = CountVectorizer(analyzer='word',
                encoding="utf-8",
                max_features=1000,
                min_df=0.0001,
                max_df=0.9)
            pos_vectorizer.fit([cd for i,cd in enumerate(clean_data) if clf[i]])
            pos_words = pos_vectorizer.get_feature_names()
            pos_features = pos_vectorizer.transform([cd for i,cd in enumerate(clean_data) if clf[i]]).toarray()
            pos_features = pos_features>0
            
            neg_vectorizer = CountVectorizer(analyzer='word',
                encoding="utf-8",
                max_features=1000,
                min_df=0.0001,
                max_df=0.9)
            neg_vectorizer.fit([cd for i,cd in enumerate(clean_data) if not(clf[i])])
            neg_words = neg_vectorizer.get_feature_names()
            neg_features = neg_vectorizer.transform([cd for i,cd in enumerate(clean_data) if not(clf[i])]).toarray()
            neg_features = neg_features>0

            shared_words = [pw for pw in pos_words if(pw in neg_words)]

            print("pos_words: %d" %(len(pos_words),))
            print("pos_docs: %d" %(sum(clf==1),))
            print("neg_words: %d" %(len(neg_words),))
            print("neg_docs: %d" %(sum(clf==0),))
            print("shared_words: %d" %(len(shared_words),))

            pos_ziped = zip(pos_words,np.asarray(pos_features.sum(axis=0)).ravel())
            sorted_pos = sorted(pos_ziped,key=lambda idx: idx[0],reverse=True)

            neg_ziped = zip(neg_words,np.asarray(neg_features.sum(axis=0)).ravel())
            sorted_neg = sorted(neg_ziped,key=lambda idx: idx[0],reverse=True)

            j = 0
            print("Shared")
            for i in range(len(sorted_pos)):
                word = sorted_pos[i][0]
                if(word in shared_words):
                    while(sorted_neg[j][0]!=word):
                        j+=1
                    pos_perc = float(sorted_pos[i][1])/sum(clf==1)
                    neg_perc = float(sorted_neg[j][1])/sum(clf==0)
                    rel = max(pos_perc,neg_perc)/min(pos_perc,neg_perc)
                    if(rel>10 and (sorted_pos[i][1]+sorted_neg[j][1])>8):
                        print("%35s\t%.3f\t%.3f\t%.3f\t%d\t%d"%(word,
                            pos_perc,
                            neg_perc,
                            rel,
                            sorted_pos[i][1],
                            sorted_neg[j][1]))

            pos_ziped = zip(pos_words,np.asarray(pos_features.sum(axis=0)).ravel())
            sorted_pos = sorted(pos_ziped,key=lambda idx: idx[1],reverse=True)
            print("pos")
            print(len(sorted_pos))
            for i in range(len(sorted_pos)):
                word = sorted_pos[i][0]
                if(not(word in shared_words)):
                    pos_perc = float(sorted_pos[i][1])/sum(clf==1)
                    print("%35s\t%.3f\t%d"%(word,
                        pos_perc,
                        sorted_pos[i][1]))
                    if(pos_perc<0.007):
                        break

            neg_ziped = zip(neg_words,np.asarray(neg_features.sum(axis=0)).ravel())
            sorted_neg = sorted(neg_ziped,key=lambda idx: idx[1],reverse=True)
            print("neg")
            print(len(sorted_neg))
            for i in range(len(sorted_neg)):
                word = sorted_neg[i][0]
                if(not(word in shared_words)):
                    neg_perc = float(sorted_neg[i][1])/sum(clf==0)
                    print("%35s\t%.3f\t%d"%(word,
                        neg_perc,
                        sorted_neg[i][1]))
                    if(neg_perc<0.004):
                        break

    def predict_probs(self, doc_ids, classes, classifier="log_reg"):
        if(self.data_origin == "csvmeta"):
            clean_data, clf = load_data_csvmeta(doc_ids,classes,self.name)
        elif(self.data_origin == "pdfinfo"):
            clean_data, clf = load_data_pdfinfo(doc_ids,classes,self.name)
        else:
            clean_data, clf = load_data_pdfcontent(doc_ids,classes,num_pages=1)

        clf = np.array(clf)

        train_data_featues = self.vectorizer.transform(clean_data).toarray()

        if(classifier=="log_reg"):
            probs = self.model.predict_proba(train_data_featues)[:,1]
        elif(classifier=="forest"):
            probs = self.model.predict_proba(train_data_featues)[:,1]
        elif(classifier=="custom_words_val"):
            probs = np.zeros(len(train_data_featues))+0.5
            probs += (np.sum(train_data_featues[:,0:self.model],axis=1)>0)*0.5
            probs -= (np.sum(train_data_featues[:,self.model:],axis=1)>0)*0.5
        elif(classifier=="custom_words_vec"):
            probs = train_data_featues>0
        else:
            print("<classifier> has to be one of [log_reg, forest, custom]. It is %s!!!"%(classifier,))
            sys.exit(1)

        return probs, clf

    def crossvalidate_text(self, doc_ids, labels, n_folds=10):

        seed=7
        np.random.seed(seed)

        print("Crossvalidating " + self.name+":")
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        iteration = 0

        for train, test in kfold.split(doc_ids,labels):

            iteration += 1
            # split the data
            train_data = [doc_ids[t] for t in train]
            test_data = [doc_ids[t] for t in test]
            train_labels = [labels[t] for t in train]
            test_labels = [labels[t] for t in test]

            classifiers = ["log_reg"]
            for  i,classifier in enumerate(classifiers):
                print("%s: iteration %d/%d"%(classifier, iteration, n_folds))
                # get the model
                self.train(train_data, train_labels, classifier)
                continue

                # scores = [[]*len(self.vectorizer.get_feature_names())]
                # scores2 = [[]*len(self.vectorizer.get_feature_names())]
                # predicted = [[]*len(self.vectorizer.get_feature_names())]

                probs_all, clf = self.predict_probs(test_data, test_labels, classifier)
                # predict
                for p in range(0,len(self.vectorizer.get_feature_names())):
                    print(self.vectorizer.get_feature_names()[p])
                    probs = probs_all[:,p]
                    preds = probs>=0.5
                    score = float(np.sum([preds==clf]))/len(clf)
                    print("accuracy: %.4f" %(score,))
                    print(confusion_matrix(clf, preds))
                pause()
                    # scores[p].append(score)
                    # scores2[p].append(score2)
                    # predicted[p].append(len(clf2))

                # for j in range(len(self.vectorizer.get_feature_names())):
                #     print("overall accuracy %.4f" %(np.mean(scores[j])))
                #     print("overall accuracy2 %.4f" %(np.mean(scores2[j])))
                #     print("overall predicted on %d/%d" %(np.mean(predicted[j]),len(clf)))

        # print("Overfitting result:")
        # self.train(doc_ids, labels)
        # self.predict_probs(doc_ids, labels, False)
        # print('\n')

    def crossvalidate(self, doc_ids, labels, n_folds=10):

        seed=7
        np.random.seed(seed)

        print("Crossvalidating " + self.name+":")
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        iteration = 0

        scores = [[],[],[],[]]
        scores2 = [[],[],[],[]]
        predicted = [[],[],[],[]]

        for train, test in kfold.split(doc_ids,labels):

            iteration += 1
            # split the data
            train_data = [doc_ids[t] for t in train]
            test_data = [doc_ids[t] for t in test]
            train_labels = [labels[t] for t in train]
            test_labels = [labels[t] for t in test]

            classifiers = ["custom_words_val"]
            for  i,classifier in enumerate(classifiers):
                print("%s: iteration %d/%d"%(classifier, iteration, n_folds))
                # get the model
                self.train(train_data, train_labels, classifier)
                # predict
                probs, clf = self.predict_probs(test_data, test_labels, classifier)
                preds = probs>=0.5
                score = float(np.sum([preds==clf]))/len(clf)
                print("accuracy: %.4f" %(score,))
                pos_dev = np.sum(np.abs(probs-0.5)[preds==clf])/np.sum([preds==clf])
                neg_dev = np.sum(np.abs(probs-0.5)[preds!=clf])/np.sum([preds!=clf])
                print("correct_std_0.5: %.4f" %(pos_dev,))
                print("flase_std_0.5: %.4f" %(neg_dev,))
                preds = probs[probs!=0.5]
                clf2 = clf[probs!=0.5]
                preds = preds>=0.5
                score2 = float(np.sum([preds==clf2]))/len(clf2)
                print("accuracy2: %.4f" %(score2,))
                print("predicted: %d/%d" %(len(clf2),len(clf)))
                print('\n')
                scores[i].append(score)
                scores2[i].append(score2)
                predicted[i].append(len(clf2))

        for j in range(len(classifiers)):
            print("%s overall accuracy on %s: %.4f" %(classifiers[j], self.name, np.mean(scores[j])))
            print("%s overall accuracy2 on %s: %.4f" %(classifiers[j], self.name, np.mean(scores2[j])))
            print("%s overall predicted on %s: %d/%d" %(classifiers[j], self.name, np.mean(predicted[j]),len(clf)))

        # print("Overfitting result:")
        # self.train(doc_ids, labels)
        # self.predict_probs(doc_ids, labels, False)
        # print('\n')

def generate_clean_training_data_metadata(field_name):
    metadata=pd.read_csv(join(DATA_PATH,"classified_metadata.csv"), delimiter=',', quoting=1, encoding='utf-8')
    clf = pd.read_csv(join(DATA_PATH,'trimmed_classification.csv'), delimiter=';', quoting=1, encoding='utf-8')

    metadata=metadata.set_index(['document_id'])
    clf=clf.set_index(['document_id'])

    print("Generating clean training files for "+field_name)

    train = metadata.loc[clf.index]
    clean_train_data = clean_metadata(field_name, train)

    if not os.path.exists(join(DATA_PATH,"lib_bow/")):
        os.makedirs(join(DATA_PATH,'lib_bow/'))

    json_path = join(DATA_PATH,'lib_bow/clean_'+field_name+'.json')

    with open(json_path, 'w') as fp:
        json.dump(clean_train_data, fp, indent=4)

def generate_clean_training_data_pdfinfo(files):
    clf = pd.read_csv(join(DATA_PATH,'trimmed_classification.csv'), delimiter=';', quoting=1, encoding='utf-8')

    clf=clf.set_index(['document_id'])
    
    existing_files=[]
    for f in files:
        if splitext(basename(f))[0] in clf.index:
            existing_files.append(f)
    
    output_dict = {}
    for i,f in enumerate(existing_files):
        meta_dict = self.pdfinfo_get_pdfmeta(f)
        output_dict[splitext(basename(f))[0]] = meta_dict

    json_path = join(MOD_PATH, 'pdfmetainfo.json')
                    
    with open(json_path, 'w') as fp:
        json.dump(output_dict, fp, indent=4)

def clean_metadata(index,data):
    clean_data={}
    number_documents=len(data.index)

    for i in range(0,number_documents):
        d_id = data.index[i]
        clean_data[d_id] = clean_csv_input(data.loc[d_id][index])
    return clean_data

def clean_csv_input(text, lang=['german','english']):
    # data is of the format: data[csv-column-name][number of row/document] - ex. data['title'][0]
    if(text is None):
        return""
    else:
        words = find_regex(text, regex=r'(?u)\b\w\w\w+\b')
        words = remove_stopwords(words)
        return " ".join(words)

def clean_meta_input(text):
    if(text is None):
        return"None"
    else:
        text = text.lower()
        clean_text = "".join(re.findall("[a-z]{2,}",text))
        text = clean_string_regex(text, regex='[^a-z]', sub="")
        return clean_text

def clean_string_regex(txt, regex=';|-|\.|,|\"|[0-9]', sub=""):
    txt = txt.lower()
    txt = re.sub(regex, sub, txt)
    return txt

def remove_whitespace(txt):
    txt = re.sub("\s", " ", txt)
    return txt

def find_regex(txt, regex=r'(?u)\b\w\w\w+\b|©'):
    words = re.findall(regex,txt)
    return words

def get_lang(txt, get_prob=False):
    if(get_prob):
        langs = detect_langs(txt)
        language = langs[0]
        return language.lang, language.prob
    else:
        language = detect(txt)
        return language

def remove_stopwords(words):
    languages = ["english", "german", "french"]
    for language in languages:
        stop_words=set(stopwords.words(language))
        words=[w for w in words if not w in stop_words]
    return words

def lemmatizer(txt, taggerdir, taggerlang):
    pass
    # tt = ttwp.TreeTagger(TAGDIR=taggerdir, TAGLANG=taggerlang)
    # taglist = tt.tag_text(txt)
    # lemmalist = []
    # for tag in taglist:
    #     lemmalist.append(tag.split('\t')[2])
    # return lemmalist

def pdfinfo_get_pdfmeta(fp):
    output = Popen(["pdfinfo", fp], stdout=PIPE, stderr=PIPE).communicate()[0].decode(errors='ignore')
    if(output==""):
        return None
    meta_dict = {}
    lines = output.split('\n')[:-1]
    new_lines=[]
    for l, line in enumerate(lines):
        if ':' in line:
            new_lines.append(line)
        else:
            new_lines[-1] += ' '+line

    for line in new_lines:
        key, val = line.split(':',1)
        key = key.lower().replace(" ", "_")
        try:
            val = val.split(None,0)[0]
        except:
            meta_dict[key] = None 
            continue
        if(key == "page_size"):
            val = val.split()
            val = [float(val[0]), float(val[2])]
        elif(key == "pages"):
            val = int(val)
        elif(key == "file_size"):
            val = val.split()[0]
            val = float(val)/1000
        elif(key == "page_rot"):
            val = int(val)>0
        elif(key == "encrypted"):
            val = not(val=="no")
        meta_dict[key] = val
    
    if not 'author' in meta_dict:
        meta_dict['author']= None
    if not 'creator' in meta_dict:
        meta_dict['creator']= None
    if not 'producer' in meta_dict:
        meta_dict['producer']= None
    if not 'title' in meta_dict:
        meta_dict['title']= None
        
    return meta_dict

def load_data_csvmeta(doc_ids,classes, name):
    clean_data = []
    clf = []
    # clean metadata is stored in lib_bow/clean_+<self.name>+.json"
    filepath = join(DATA_PATH,"lib_bow/clean_"+name+".json")
    # the data is stored as a dict{doc_id:string}
    data = json.load(open(filepath,"r"))

    # go through doc_ids and add the text
    for d_id, d_cls in zip(doc_ids, classes):
        if(d_id in data):
            clean_data.append(data[d_id])
            clf.append(d_cls)

    return clean_data, clf

def load_data_pdfinfo(doc_ids,classes,field):
    clean_data = []
    clf = []
    # pdfinfo data is stored in lib_bow/pdfmetainfo.json
    filepath = join(DATA_PATH,"lib_bow/pdfmetainfo.json")
    # the data is stored as a dict of dicts {doc_id:{key_string:string}}
    data = json.load(open(filepath,"r"))

    # go through doc_ids and add the text
    for d_id, d_cls in zip(doc_ids, classes):
        if(d_id in data):
            clean_test_data = ""
            input_dict = data[d_id]
            # if the dict at the id is non the file was password protected
            if(input_dict is None):
                clean_test_data = "passwordprotected"
            else:
                input_string = input_dict[field]
                # if the string for a position is None this field was empty
                if(input_string is None):
                    clean_test_data = "None"
                else:
                    # data is not cleaned yet
                    clean_test_data = clean_meta_input(input_string)
            clean_data.append(clean_test_data)
            clf.append(d_cls)

    return clean_data, clf

def load_data_pdfcontent(doc_ids,classes,num_pages):
    clean_data = []
    clf = []
    empty_pos = 0
    empty_neg = 0
    # test is stored in a seperate json per pdf
    for d_id, d_cls in zip(doc_ids, classes):
        json_path = join(TXT_PATH,d_id+".json")
        if(isfile(json_path)):
            # data in the json is a dict{page:string}
            data = json.load(open(json_path,"r"))
            # if the dict is None the document was password protected
            if(data is None):
                test_data = "password_protected"
            elif(len(data)==0):
                test_data = "None"
            else:
                l_page = max(list(map(int, data.keys())))
                test_data = ""
                # concatenate the text of maximal the specified num_pages
                for i in range(1,max(l_page,num_pages)+1):
                    test_data += data[str(i)]
                # text is not cleaned yet, clean it
                clean_test_data = clean_pdf_txt_content(test_data)
                # clean_test_data = test_data
            if(len(clean_test_data)>20):
                clean_data.append(clean_test_data)
                clf.append(d_cls)
            # # since the text can be very large give a Warning if it gets really high
            # if(sys.getsizeof(clean_data)>1000000000):
            #     print("Warning!!! Input data is larger than 1GB!")
        else:
            print(json_path + " is no valid file!!")

    return clean_data, clf

def clean_pdf_txt_content(txt):
    txt = remove_whitespace(txt)
    # words = find_regex(txt)
    words = remove_stopwords(txt.split())
    txt =  " ".join(words)
    return txt



if __name__ == "__main__":

    sys.path.append("/home/kai/Workspace/deep_doc_class/deep_doc_class/src")
    from doc_globals import*
    # features = ["title", "filename", "folder_name", "folder_description", "description"]
    # for field_name in features:
    #     generate_clean_training_data_metadata(field_name)

    args = sys.argv
    train_file = args[1]
    train = pd.read_csv(train_file, delimiter=',', header=0, quoting=1)
    train.columns = ["class", "document_id"]
    doc_ids = list(train["document_id"])
    classes = list(train["class"])

    features = []
    # # # features.append(BowClassifier("title"))
    # # # features.append(BowClassifier("folder_description"))
    # # # features.append(BowClassifier("description"))
    # # # features.append(BowClassifier("author"))

    # features.append(BowClassifier("filename"))
    # features.append(BowClassifier("folder_name"))
    # features.append(BowClassifier("creator"))
    # features.append(BowClassifier("producer"))
    features.append(BowClassifier("text"))

    # for f in features:
    #     f.train(doc_ids, classes)
    #     print(f.name[0])

    for f in features:
        f.crossvalidate_text(doc_ids, classes, 10)
        pause()