# -*- coding: utf-8 -*-

"""
This class tries to predicted if a pdf file is copy protected by extracting features that
could suggest a document is scanned. The features are composed by the producer field
of the metadata and the extraction of the semantic regions of the file using an 
external program called pdf-extract. If the regions fail to be extracted it is assumed the
file is scanned.
@author: Renato Garita Figueiredo
"""
import csv, os, shutil, sys
import numpy as np
from subprocess import Popen, PIPE
from PyPDF2 import PdfFileReader, PdfFileWriter, utils
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import Imputer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from os.path import dirname, join, realpath
MOD_PATH = dirname(realpath(__file__))

class Meta_Info:
    
    def __init__(self): 
        
        self.name = ['meta_info_author','meta_info_creator','meta_info_producer', 'encrypted']

        if os.path.isfile(join(MOD_PATH,'author-trained-naive.pkl')) and os.path.isfile(join(MOD_PATH,'author-vectorizer.pkl')) and os.path.isfile(join(MOD_PATH,'creator-trained-naive.pkl')) and os.path.isfile(join(MOD_PATH,'creator-vectorizer.pkl')) and os.path.isfile(join(MOD_PATH,'producer-trained-naive.pkl')) and os.path.isfile(join(MOD_PATH,'producer-vectorizer.pkl')):
            self.gnl_auth = joblib.load(join(MOD_PATH,'author-trained-naive.pkl'))
            self.vec_auth = joblib.load(join(MOD_PATH,'author-vectorizer.pkl'))
            self.gnl_crea = joblib.load(join(MOD_PATH,'creator-trained-naive.pkl'))
            self.vec_crea = joblib.load(join(MOD_PATH,'creator-vectorizer.pkl'))
            self.gnl_prod = joblib.load(join(MOD_PATH,'producer-trained-naive.pkl'))
            self.vec_prod = joblib.load(join(MOD_PATH,'producer-vectorizer.pkl'))
        else:
            args = sys.argv
            len_args = len(args)
            
            usage = "python meta_info_module.py <files_path> <files_classes_file>"

            if len_args == 3:
                self.train(args[1], args[2])
                print("Training done!")
            else:
                print("No trained model or vectorizer found.")
                print("Usage:",usage)

    #@param inputpdf a pointer to a PdfFileReader object
    #
    #This function reads the metadata information of a pdf file and extracts the
    #author, creator and producer fields. Some copyrighted documents have common fields based on the 
    #machine they were scanned with. Other values like Powerpoint can suggest 
    #the file is a presentation and, therefore, not copy protected
    def readMetadata(self, file):
        
        author_feature={}
        creator_feature={}
        producer_feature={}
        encrypted_feature={}
                                            
        inputpdf = PdfFileReader(file,strict=False)
                
        if not inputpdf.isEncrypted:
        
            encrypted_feature = 0.0
                
            pdf_info = inputpdf.getDocumentInfo()
                
            if isinstance(pdf_info,dict):
                
                if '/Author' in pdf_info:
                    author = ''.join(i for i in pdf_info['/Author'] if not str(i).isdigit())
                    author = ''.join(i for i in author if not i == ".")
                    author = ''.join(i for i in author if not i == ",")
                    author = ''.join(i for i in author if not i == "-")
                    author = ''.join(i for i in author if not i == ";")
                    author = ''.join(i for i in author if not i == " ")
                    
                    author_feature['author']=author
                else:
                    author_feature['author']='null'
    
                if '/Creator' in pdf_info:
                    creator = ''.join(i for i in pdf_info['/Creator'] if not str(i).isdigit())
                    creator = ''.join(i for i in creator if not i == ".")
                    creator = ''.join(i for i in creator if not i == ",")
                    creator = ''.join(i for i in creator if not i == "-")
                    creator = ''.join(i for i in creator if not i == ";")
                    creator = ''.join(i for i in creator if not i == " ")
                    
                    creator_feature['creator']=creator
                else:
                    creator_feature['creator']='null'
    
                if '/Producer' in pdf_info:
                    producer = ''.join(i for i in pdf_info['/Producer'] if not str(i).isdigit())
                    producer = ''.join(i for i in producer if not i == ".")
                    producer = ''.join(i for i in producer if not i == ",")
                    producer = ''.join(i for i in producer if not i == "-")
                    producer = ''.join(i for i in producer if not i == ";")
                    producer = ''.join(i for i in producer if not i == " ")
                    
                    producer_feature['producer']=producer
                else:
                    producer_feature['producer']='null'
            else:
                author_feature['author']='null'
                creator_feature['creator']='null'
                producer_feature['producer']='null'
        else:
            
            encrypted_feature = 1.0

            print("Encrypted file. Can't extract features.",file)
            author_feature['author']='Encrypted'
            creator_feature['creator']='Encrypted'
            producer_feature['producer']='Encrypted'
            
            #shutil.move(root+'/'+file, root+'/Encrypted/'+file)
                
                                                
        return author_feature, creator_feature, producer_feature, encrypted_feature
        
        
    def classify(self,filepointer):
                
        #Extract features
        fa, fc, fp, fe = self.readMetadata(filepointer)
                    
        #Transform categorical data to numerical (binary representation)
        feature_list_trans_auth = self.vec_auth.transform(fa).toarray()
        feature_list_trans_crea = self.vec_crea.transform(fc).toarray()
        feature_list_trans_prod = self.vec_prod.transform(fp).toarray()
                
        return self.gnl_auth.predict(feature_list_trans_auth)[0], self.gnl_crea.predict(feature_list_trans_crea)[0], self.gnl_prod.predict(feature_list_trans_prod)[0], fe
        
    #@param filepointer a pointer to a pdf file
    #@param metapointer a pointer to the metadata, this parameter is not used
    #@return float64 [0 1] probabiliy for the pdf  beeing copyright protected     
    def get_function(self,filepointer, metapointer = None):
        
        try:
            result = []
            fa, fc, fp, fe = self.classify(filepointer)
            result.append(fa)
            result.append(fc)
            result.append(fp)
            result.append(fe)
            return result
        except:
            return [np.nan, np.nan, np.nan, np.nan]
        
    #@param path the path where the training data is located
    #@param class_path the path to where the classification.csv is located
    #@param num_pages the amount of pages to process starting from the first (default 1)
    def train(self,files_path,class_path,num_pages=1):
        
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
                    #shutil.move(files_path+'/'+file+'.pdf', files_path+'/Damaged/'+file+'.pdf')
 
            except:
                print("Unexpected error",file+'.pdf')
                #shutil.move(files_path+'/'+file+'.pdf', files_path+'/Damaged/'+file+'.pdf')
                exit()


               
        vec=DictVectorizer()

        vec.fit(feat_author)
        fts_trans=vec.transform(feat_author).toarray()
        imp = Imputer(missing_values='NaN',strategy='most_frequent',axis=0)
        FTS=imp.fit_transform(fts_trans)
               
        gnb = MultinomialNB()
        gnb.fit(FTS,classes)
        
#        print(gnb.coef_)
#        print(gnb.score(FTS, classes))
#        print(gnb.predict(FTS))
                        
        joblib.dump(vec, join(MOD_PATH,'author-vectorizer.pkl'))        
        joblib.dump(gnb, join(MOD_PATH,'author-trained-naive.pkl'))    
        
        vec=DictVectorizer()

        vec.fit(feat_creator)
        fts_trans=vec.transform(feat_creator).toarray()
        imp = Imputer(missing_values='NaN',strategy='most_frequent',axis=0)
        FTS=imp.fit_transform(fts_trans)
               
        gnb = MultinomialNB()
        gnb.fit(FTS,classes)
        
#        print(gnb.coef_)
#        print(gnb.score(FTS, classes))
#        print(gnb.predict(FTS))
                        
        joblib.dump(vec, join(MOD_PATH,'creator-vectorizer.pkl'))       
        joblib.dump(gnb, join(MOD_PATH,'creator-trained-naive.pkl'))
        
        vec=DictVectorizer()

        vec.fit(feat_producer)
        fts_trans=vec.transform(feat_producer).toarray()
        imp = Imputer(missing_values='NaN',strategy='most_frequent',axis=0)
        FTS=imp.fit_transform(fts_trans)
               
        gnb = MultinomialNB()
        gnb.fit(FTS,classes)
        
#        print(gnb.coef_)
#        print(gnb.score(FTS, classes))
#        print(gnb.predict(FTS))
                        
        joblib.dump(vec, join(MOD_PATH,'producer-vectorizer.pkl'))       
        joblib.dump(gnb, join(MOD_PATH,'producer-trained-naive.pkl'))
        

    
#Meta_Info()