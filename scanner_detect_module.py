# -*- coding: utf-8 -*-

"""
This class tries to predicted if a pdf file is copy protected by extracting features that
could suggest a document is scanned. The features are composed by the producer field
of the metadata and the extraction of the semantic regions of the file using an 
external program called pdf-extract. If the regions fail to be extracted it is assumed the
file is scanned.

@author: Renato Garita Figueiredo
"""
import csv, os, shutil
import numpy as np
from subprocess import Popen, PIPE
from PyPDF2 import PdfFileReader, PdfFileWriter, utils
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import Imputer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib

class ScannerDetect:
    
    def __init__(self): 
        if os.path.isfile('scanner-trained-naive.pkl') and os.path.isfile('scanner-vectorizer.pkl'):
            self.gnl = joblib.load('scanner-trained-naive.pkl')
            self.vec = joblib.load('scanner-vectorizer.pkl')
        else:
            print("No trained model or vectorizer found.")

    #@param inputpdf a pointer to a PdfFileReader object
    #
    #This function reads the metadata information of a pdf file and extracts the
    #producer field. Some scanned documents have common producers based on the 
    #machine they were scanned with. Other producers like Powerpoint can suggest 
    #the file is a presentation and, therefore, not copy protected
    def readMetadata(self,inputpdf):
        
        file_features={}
                        
        pdf_info = inputpdf.getDocumentInfo()
    
        if isinstance(pdf_info,dict):
            if '/Producer' in pdf_info:
                producer = ''.join(i for i in pdf_info['/Producer'] if not i.isdigit())
                producer = ''.join(i for i in producer if not i == ".")
                producer = ''.join(i for i in producer if not i == ",")
                producer = ''.join(i for i in producer if not i == "-")
                producer = ''.join(i for i in producer if not i == ";")
                producer = ''.join(i for i in producer if not i == " ")
                
                file_features['producer']=producer
        else:
            file_features['producer']=np.nan
                                                
        return file_features
                        
    #@param inputpdf a pointer to a PdfFileReader object
    #@param num_pages the amount of pages to process starting from the first (default 1)  
    #
    #This function uses the external program pdf-extract to extract the regions of a pdf
    #if no regions are extracted the file is more likely to be scanned and, therefore,
    #copy protected. Don't use a lot of pages because it gets really slow, 1 page is usually
    #more than enough.
    def extractRegions(self,inputpdf,num_pages):
        
        file_features={}

        for i in range(num_pages):
            output = PdfFileWriter()
            output.addPage(inputpdf.getPage(i))
            new_file="document-page%s.pdf" % i
            with open(new_file, "wb") as outputStream:
                output.write(outputStream)
                
        line_count=0
        with Popen(["pdf-extract","extract","--no-lines","--regions",new_file], stdout=PIPE, universal_newlines=True) as process:
            for line in process.stdout:
                line_count += 1

        if line_count == 2:
            file_features['lines']='two'
        else:
            file_features['lines']='other'
            
        return file_features
        
        
    def classify(self,filepointer):
        
        feature_list=[]
        sf=rf={}
        
        try:
            inputpdf = PdfFileReader(filepointer,strict=False)
        except utils.PdfReadError:
            print("Error reading file")
            return np.nan
        
        if not inputpdf.isEncrypted:
            #Extract features
            sf=self.readMetadata(inputpdf)
            if shutil.which('pdf-extract') is not None:
                rf=self.extractRegions(inputpdf,1)
                fm={**sf,**rf}
            else:
                print("Missing pdf-extract. Only using metadata features.")
                fm=sf
        else:
            print("Encrypted file. Can't extract features.")
            return np.nan
            
        #Arrange data
        feature_list.append(fm)
        
        #Transform categorical data to numerical (binary representation)
        feature_list_trans=self.vec.transform(feature_list).toarray()
        
        #Impute missing values using a most_frequent strategy (maybe we don't want this)
        imp = Imputer(missing_values='NaN',strategy='most_frequent',axis=1)
        feature_list_imp=imp.fit_transform(feature_list_trans)
                
        return self.gnl.predict(feature_list_imp)
        
    #@param filepointer a pointer to a pdf file
    #@param metapointer a pointer to the metadata, this parameter is not used
    #@return float64 [0 1] probabiliy for the pdf  beeing copyright protected     
    def get_function(self,filepointer, metapointer = None):
        return float(self.classify(filepointer))
        
    #@param path the path where the training data is located
    #@param class_path the path to where the classification.csv is located
    #@param num_pages the amount of pages to process starting from the first (default 1)
    def train(self,files_path,class_path,num_pages=1):
        
        features=[]
        
        with open(class_path) as classcsvfile:
                classdata = csv.DictReader(classcsvfile, delimiter=';')
                data_list=list(classdata)
                class_copy=[]
            
                for root, dirs, files in os.walk(files_path):
                   for file in files:
                       
                       try:
                           inputpdf = PdfFileReader(open(root+'/'+file,'rb'),strict=False)
                            
                           if not inputpdf.isEncrypted:                                   
                               
                               file_name=os.path.splitext(file)[0]
                               for d in data_list:
                                   
                                   file_not_found=True
                                   
                                   if d['document_id'] == file_name:
                                       if d['published'] == 'True':
                                           class_copy.append(1)
                                       elif d['published'] == 'False':
                                           class_copy.append(0)
                                       else:
                                           print("Published:",d['published'])
                                   
                                       sf=self.readMetadata(inputpdf)
                                       rf=self.extractRegions(inputpdf,num_pages)
                                       fts={**sf,**rf}
                                       features.append(fts)
                                       
                                       if(len(features) != len(class_copy)):
                                           print("Something went wrong!")
                                           print(len(features))
                                           print(len(features[0]))
                                           print(len(class_copy))
                                           exit()
                                           
                                       file_not_found=False
                                       
                                       break
                                           
                                
                               if file_not_found:
                                    print("FILE NOT ON THE LIST")
                                    shutil.move(root+'/'+file, root+'/Unlisted/'+file)
                                   

                           else:
                               print("Encrypted file. Can't extract features.",file)
                               shutil.move(root+'/'+file, root+'/Encrypted/'+file)
                               
                       except utils.PdfReadError:
                            print("Error reading file",file)
                            shutil.move(root+'/'+file, root+'/Damaged/'+file)
                            
                   break

        
                vec=DictVectorizer()

                vec.fit(features)
                fts_trans=vec.transform(features).toarray()
                imp = Imputer(missing_values='NaN',strategy='most_frequent',axis=0)
                FTS=imp.fit_transform(fts_trans)
                       
                gnb = MultinomialNB()
                gnb.fit(FTS,class_copy)
                
                print(gnb.coef_)
                print(gnb.score(FTS, class_copy))
                print(gnb.predict(FTS))
                                
                joblib.dump(vec, 'scanner-vectorizer.pkl')        
                joblib.dump(gnb, 'scanner-trained-naive.pkl')        
        

    

