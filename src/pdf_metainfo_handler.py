#pdf_metainfo_handler.py

import sys, os, re, subprocess
from os.path import join, realpath, dirname, isdir, isfile, basename
MOD_PATH = dirname(realpath(__file__))
from doc_globals import* 

import numpy as np
from collections import Counter

from sklearn.externals import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import Imputer
from sklearn.naive_bayes import MultinomialNB


from time import time
import json, csv
import pandas as pd

from PyPDF2 import PdfFileWriter, PdfFileReader, utils
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument

try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO

class Meta_PDF_Module:

    def __init__(self):
        # mats metainfo

        bow_file = join(MOD_PATH,'meta_extractor_bows.txt')
        if(isfile(bow_file)):
            with open(bow_file,'r') as save:
                self.lib_list = eval(save.read())
        else:
            print("meta info mats is not trained!!\n Train the module first!!")
            sys.exit(1)

        self.exclude = exclude
        self.name = ['pos_author','pos_creator','pos_producer','neg_author','neg_creator','neg_producer',
        'meta_info_author','meta_info_creator','meta_info_producer', 'encrypted',
        'rotated', 'width', 'height', 'filesize_ratio']

        # renato metainfo
        a_vec = join(MOD_PATH,'author-vectorizer.pkl')
        a_train = join(MOD_PATH,'author-trained-naive.pkl')
        c_vec = join(MOD_PATH,'creator-vectorizer.pkl')
        c_train = join(MOD_PATH,'creator-trained-naive.pkl')
        p_vec = join(MOD_PATH,'producer-vectorizer.pkl')
        p_train = join(MOD_PATH,'producer-trained-naive.pkl')

        if(isfile(a_train) and isfile(a_vec) and isfile(c_train) and isfile(c_vec) and isfile(p_train) and isfile(p_vec)):
            self.gnl_auth = joblib.load(a_train)
            self.vec_auth = joblib.load(a_vec)
            self.gnl_crea = joblib.load(c_train)
            self.vec_crea = joblib.load(c_vec)
            self.gnl_prod = joblib.load(p_train)
            self.vec_prod = joblib.load(p_vec)

        else:
            print("meta info renato is not trained!!\n Train the module first!!")
            sys.exit(1)

    def pdfinfo_get_pdfmeta(fp):
        output = subprocess.Popen(["pdfinfo", fp], stdout=subprocess.PIPE).communicate()[0].decode()
        if(output==""):
            return None
        meta_dict = {}
        lines = output.split('\n')[:-1]
        for line in lines:
            key, val = line.split(':',1)
            print(key + "\t" + val)
            key = key.lower().replace(" ", "_")
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
            
        return meta_dict

    def get_function(self,fp):

        author_feature={}
        creator_feature={}
        producer_feature={}
        encrypted_feature={}

        result = {}
        meta_dict = pdfinfo_get_pdfmeta(fp)

        if(meta_dict is None):
            author_feature['author']='null'
            creator_feature['creator']='null'
            producer_feature['producer']='null'

            result = [np.nan]*len(self.name)
            return result

        else:
            author = meta_dict['author']
            pos, neg = self.get_bow_score(self.get_bow(author),0)
            result['pos_author'] = pos
            result['neg_author'] = neg

            author = self.clean_string(self, author, regex='\s|;|-|\.|,|[0-9]')
            author_feature['author'] = author

            creator = meta_dict['creator']
            pos, neg = self.get_bow_score(self.get_bow(creator),1)
            result['pos_creator'] = pos
            result['neg_creator'] = neg

            creator = self.clean_string(self, creator, regex='\s|;|-|\.|,|[0-9]')
            creator_feature['author'] = creator

            producer = meta_dict['producer']
            pos, neg = self.get_bow_score(self.get_bow(producer),2)
            result['pos_producer'] = pos
            result['neg_producer'] = neg

            producer = self.clean_string(self, producer, regex='\s|;|-|\.|,|[0-9]')
            producer_feature['author'] = producer

            result['rotated'] = meta_dict['page_rot']

            result['width'] = meta_dict['page_size'][0]
            result['height'] = meta_dict['page_size'][0]

            result['filesize_ratio'] = float(meta_dict['file_size'])/meta_dict['pages']

            if(meta_dict['encrypted']):
                author_feature['author'] = "Encrypted"
                creator_feature['author'] = "Encrypted"
                producer_feature['author'] = "Encrypted"

                result['encrypted'] = 1.0
            else:
                result['encrypted'] = 0.0

            feature_list_trans_auth = self.vec_auth.transform(author_feature).toarray()
            feature_list_trans_crea = self.vec_crea.transform(creator_feature).toarray()
            feature_list_trans_prod = self.vec_prod.transform(producer_feature).toarray()

            result['meta_info_author'] = self.gnl_auth.predict(feature_list_trans_auth)[0]
            result['meta_info_creator'] = self.gnl_crea.predict(feature_list_trans_crea)[0]
            result['meta_info_producer'] = self.gnl_prod.predict(feature_list_trans_prod)[0]
        
            return result

    def clean_string(self, txt, regex=';|-|\.|,|[0-9]'):
        txt = txt.lower()
        re.sub(regex, "", txt)
        return txt

    def get_bow(self,txt):
        """
        @param txt:     the pdf-content as a string
        @return:        the bow of that specific document
        """
        txt = self.clean_string(txt, regex=';|-|\.|,|[0-9]')
        return Counter(re.findall(r'\w+', txt))

    def get_bow_score(self,bow,index):
        """
        :param bow: current text as bow
        :param index: 1 = author
                      2 = creator
                      3 = producer
        :return:    positive and negative score (float)
        """
        pos = 0.0
        neg = 0.0

        for key in bow:
            if key in self.lib_list[index]:
                pos += bow[key]*self.lib_list[index][key]
            if key in self.lib_list[index+3]:
                neg += bow[key]*self.lib_list[index+3][key]
                
        return pos,neg

    def train(self,filenames,classes):
        pos_creators = list()
        pos_authors = list()
        pos_producer = list()
        neg_creators = list()
        neg_authors = list()
        neg_producer = list()
        for i in range(len(filenames)):
            f = filenames[i]
            if classes[i] == 'True':
                try:
                    try:
                        mp = self.get_meta(open(join(PDF_PATH,f),'rb'))
                    except:
                        continue
                    if('/Creator' in mp):
                        pos_creators.append(self.get_bow(mp['/Creator']))
                    if '/Author' in mp:
                        pos_authors.append(self.get_bow(mp['/Author']))
                    if '/Producer' in mp:
                        pos_producer.append(self.get_bow(mp['/Producer']))
                except:
                    continue
            else:
                try:
                    try:
                        mp = self.get_meta(open(join(PDF_PATH,f),'rb'))
                    except:
                        continue
                    if('/Creator' in mp):
                        neg_creators.append(self.get_bow(mp['/Creator']))
                    else:
                        neg_creators.append(Counter({'null': 1}))
                    if '/Author' in mp:
                        neg_authors.append(self.get_bow(mp['/Author']))
                    else:
                        neg_authors.append(Counter({'null': 1}))
                    if '/Producer' in mp:
                        neg_producer.append(self.get_bow(mp['/Producer']))
                    else:
                        neg_producer.append(Counter({'null': 1}))
                except:
                    continue
        #sum up
        pos_authors = sum(pos_authors,Counter())
        pos_creators = sum(pos_creators,Counter())
        pos_producer = sum(pos_producer,Counter())
        neg_authors = sum(neg_authors,Counter())
        neg_creators = sum(neg_creators,Counter())
        neg_producer = sum(neg_producer,Counter())

        lib_list = list()

        lib_list.append(self.normalize_bow(pos_authors))
        lib_list.append(self.normalize_bow(pos_creators))
        lib_list.append(self.normalize_bow(pos_producer))
        lib_list.append(self.normalize_bow(neg_authors))
        lib_list.append(self.normalize_bow(neg_creators))
        lib_list.append(self.normalize_bow(neg_producer))

        with open(join(MOD_PATH,'meta_extractor_bows.txt'),'w+') as save:
            save.write(str(lib_list))
        self.lib_list = lib_list
        return


    def normalize_bow(self,counter):
        total = 0
        for keys in counter:
            total += counter[keys]
        for key in counter:
            counter[key] /= total
        return counter

    #@param path the path where the training data is located
    #@param class_path the path to where the classification.csv is located
    #@param num_pages the amount of pages to process starting from the first (default 1)
    def train2(self,files_path,class_path,num_pages=1):
        
        feat_author=[]
        feat_creator=[]
        feat_producer=[]
        
        with open(class_path, 'r') as classcsvfile:
            classdata = csv.reader(classcsvfile)
            data_list=list(classdata)
            
        files = [x[1] for x in data_list]
        classes = [x[0] for x in data_list]

        for file in files:
            
            try:
                pdf_file = join(files_path,file+'.pdf')
                if not os.stat(pdf_file).st_size == 0:

                    fa, fc, fp = self.readMetadata(open(pdf_file,'rb'))
                    feat_author.append(fa)
                    feat_creator.append(fc)
                    feat_producer.append(fp)
                
                else:
                    print("File is 0 bytes",file)
                    fa={np.nan}
                    fc={np.nan}
                    fp={np.nan}
 
            except:
                print("Unexpected error",file+'.pdf')
                exit()
               
        vec=DictVectorizer()

        vec.fit(feat_author)
        fts_trans=vec.transform(feat_author).toarray()
        imp = Imputer(missing_values='NaN',strategy='most_frequent',axis=0)
        FTS=imp.fit_transform(fts_trans)
               
        gnb = MultinomialNB()
        gnb.fit(FTS,classes)
                        
        joblib.dump(vec, join(MOD_PATH,'author-vectorizer.pkl'))        
        joblib.dump(gnb, join(MOD_PATH,'author-trained-naive.pkl'))    
        
        vec=DictVectorizer()

        vec.fit(feat_creator)
        fts_trans=vec.transform(feat_creator).toarray()
        imp = Imputer(missing_values='NaN',strategy='most_frequent',axis=0)
        FTS=imp.fit_transform(fts_trans)
               
        gnb = MultinomialNB()
        gnb.fit(FTS,classes)
                        
        joblib.dump(vec, join(MOD_PATH,'creator-vectorizer.pkl'))       
        joblib.dump(gnb, join(MOD_PATH,'creator-trained-naive.pkl'))
        
        vec=DictVectorizer()

        vec.fit(feat_producer)
        fts_trans=vec.transform(feat_producer).toarray()
        imp = Imputer(missing_values='NaN',strategy='most_frequent',axis=0)
        FTS=imp.fit_transform(fts_trans)
               
        gnb = MultinomialNB()
        gnb.fit(FTS,classes)
                        
        joblib.dump(vec, join(MOD_PATH,'producer-vectorizer.pkl'))       
        joblib.dump(gnb, join(MOD_PATH,'producer-trained-naive.pkl'))