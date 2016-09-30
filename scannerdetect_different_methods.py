# -*- coding: utf-8 -*-

"""
Spyder Editor

This is a temporary script file.
"""
import csv
import sys
import os
import numpy as np
from subprocess import Popen, PIPE
from itertools import islice
from PyPDF2 import PdfFileReader, PdfFileWriter, utils
from pdfquery import PDFQuery
from gi.repository import Poppler
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import Imputer
from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib

class ScannerDetect:
    def __init__(self):
        self.md,self.cd = self.openFiles()  
        self.scan_metadata = {}
        self.features_true = []
        self.features_false = []
        self.features = []
        
    #@param filepointer a pointer to a pdf file
    #@param metapointer a pointer to the metadata, this parameter is not used
    #@return float64 [0 1] probabiliy for the pdf  beeing copyright protected     
    def get_function(self,filepointer, metapointer = None):
        x = len(self.convert_pdf_to_txt(filepointer,-1))
        #return s.norm.pdf(self.mean,self.std,x)
        
    def readMetadata(self,file,cls):
        
        file_features={}
        
        try:
            fp = PdfFileReader(open(file, 'rb'),strict=False)
            
            file_features['producer']=np.nan
            
            if not fp.isEncrypted:
                pdf_info = fp.getDocumentInfo()
        
                if isinstance(pdf_info,dict):
                    if '/Producer' in pdf_info:
                        producer = ''.join(i for i in pdf_info['/Producer'] if not i.isdigit())
                        producer = ''.join(i for i in producer if not i == ".")
                        producer = ''.join(i for i in producer if not i == ",")
                        producer = ''.join(i for i in producer if not i == "-")
                        producer = ''.join(i for i in producer if not i == ";")
                        producer = ''.join(i for i in producer if not i == " ")
                        
                        file_features['producer']=producer
                        
                        if not producer in self.scan_metadata:
                            self.scan_metadata[producer] = [0,0]
    
                        if cls:
                            value_true = self.scan_metadata[producer][0]+1
                            value_false = self.scan_metadata[producer][1]
                        else:
                            value_false = self.scan_metadata[producer][1]+1
                            value_true = self.scan_metadata[producer][0]
                            
                        self.scan_metadata[producer] = [value_true,value_false]
                                                
        except utils.PdfReadError:
            file_features['producer']=np.nan
            print("Error")
                    
        sorted_true_scan_metadata = sorted(self.scan_metadata.items(), key=lambda x: x[1][0],reverse=True)
        sorted_false_scan_metadata = sorted(self.scan_metadata.items(), key=lambda x: x[1][1],reverse=True)

        return (file_features,sorted_true_scan_metadata,sorted_false_scan_metadata) 
        
    def extractPDFData(self,file):
        absfile = os.path.realpath(file)
        print(absfile)
        pdf_doc=Poppler.Document('file://'+absfile)
        print('No of pages', pdf_doc.no_of_pages)
        for p in pdf_doc:
            print('Page', p.page_no, 'size =', p.size)
            for f in p:
                print(' '*1,'Flow')
                for b in f:
                    print(' '*2,'Block', 'bbox=', b.bbox.as_tuple())
                    for l in b:
                        print(' '*3, l.text.encode('UTF-8'), '(%0.2f, %0.2f, %0.2f, %0.2f)'% l.bbox.as_tuple())
                        #assert l.char_fonts.comp_ratio < 1.0
                        for i in range(len(l.text)):
                            print(l.text[i].encode('UTF-8'), '(%0.2f, %0.2f, %0.2f, %0.2f)'% l.char_bboxes[i].as_tuple(),\
                                l.char_fonts[i].name, l.char_fonts[i].size, l.char_fonts[i].color)
                        
                        
        
    def getLayout(self,file):
        pdf = PDFQuery(file)
        print(pdf.get_layout(0))
        tree = pdf.get_tree(0)
        print(tree)
                
        
    def extractRegions(self,file,num_pages):
        
        file_features={}
        
        try:
            #print(file)
            inputpdf = PdfFileReader(open(file, "rb"))
    
            if not inputpdf.isEncrypted:
                for i in range(num_pages):
                    output = PdfFileWriter()
                    output.addPage(inputpdf.getPage(i))
                    new_file="document-page%s.pdf" % i
                    with open(new_file, "wb") as outputStream:
                        output.write(outputStream)
                        
                line_count=0
                with Popen(["pdf-extract","extract","--no-lines","--regions",new_file], stdout=PIPE, universal_newlines=True) as process:
                    for line in process.stdout:
                        #print(line)
                        line_count += 1
                        #if line_count > 2:
                        #    break
                    #read_chunk = partial(process.stdout.read, 1 << 13)
                    #line_count = sum(chunk.count(b'\n') for chunk in iter(read_chunk, b''))
                    
                #print(line_count)
                if line_count == 2:
                    file_features['lines']='two'
                else:
                    file_features['lines']='other'
                    
        except utils.PdfReadError:
            file_features['lines']=np.nan
            print("Error")
            
        return file_features

        #(output, err) = process.communicate()
        #exit_code = process.wait()
        #for line in process.stdout.splitlines:
        #    print("Output: ",line)
        
        #print("Error: ",process.stderr)
        
        #for row in metadata:
        #print(row['document_id'],row['filename'],row['title'],row['description'],row['is_pdf'],row['folder_name'],row['folder_description'],row['author_prof'],row['author_dr'],row['status'],row['upload_timestamp'],row['filesize'],row['institute'])

                



test = ScannerDetect()
    
if len(sys.argv) > 1:
    files_path = sys.argv[1]
    #num_pages = sys.argv[2]
    num_pages = 1
    
    with open('../classification.csv') as classcsvfile:
        classdata = csv.DictReader(classcsvfile, delimiter=';')
        data_list=list(classdata)
        iterator = islice(classdata,1)
        cnt = 0
        vec=DictVectorizer()
        cc={}
        class_copy=[]
    
        for root, dirs, files in os.walk(files_path+"/TrainData"):
           for file in files:
               file_name=os.path.splitext(file)[0]
               for d in data_list:
                   if d['document_id'] == file_name:
                       if d['published'] == 'True':
                           class_copy.append(1)
                       elif d['published'] == 'False':
                           class_copy.append(0)
                       break
                   
               (ftp,ss,sp)=test.readMetadata(root+"/"+file,True)
               ftl=test.extractRegions(root+"/"+file,num_pages)
               fts={**ftp,**ftl}
               test.features.append(fts)

               #cnt+=1
               #if cnt == 450:
               #    print("FINISH TRUE")
               #    break
        #imp = Imputer(missing_values='NaN',strategy='most_frequent',axis=0)
        #T=imp.fit_transform(test.features_true)
        print(class_copy)
        print(test.features)
        fts_trans=vec.fit_transform(test.features).toarray()
        #cls_trans=vec.fit_transform(class_copy).toarray()
        print(fts_trans)
        print('\n')
        imp = Imputer(missing_values='NaN',strategy='most_frequent',axis=0)
        FTS=imp.fit_transform(fts_trans)
        print('\n')
        print('Features')
        print(FTS)
        print(len(FTS))
        print('Classes')
        print(class_copy)
        print(len(class_copy))
        print(len(class_copy))
        cnt=0
        print('\n')
               
        
               
        regr = linear_model.LinearRegression()
        cls=regr.fit(FTS, class_copy)
        
        print(cls.coef_)
        
        joblib.dump(cls, 'scanner-trained-linear.pkl') 
        
        cll = joblib.load('scanner-trained-linear.pkl') 
        
        print(cll.coef_)
        
        print(cll.score(FTS, class_copy))
        print(np.mean((cll.predict(FTS) - class_copy) ** 2))
        
        gnb = MultinomialNB()
        cgn=gnb.fit(FTS,class_copy)
        
        print(gnb.coef_)
        
        joblib.dump(gnb, 'scanner-trained-naive.pkl') 
        
        gnl = joblib.load('scanner-trained-naive.pkl') 
        
        print(gnl.coef_)
        
        print(gnl.score(FTS, class_copy))
        print(gnl.predict(FTS))
        print(np.mean((gnl.predict(FTS) - class_copy) ** 2))                
        
            
else:
    print("Specify file")


#glm bias alfa=sum(features)
