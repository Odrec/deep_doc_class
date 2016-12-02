# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 19:58:54 2016

This module generates a single dimension for the datavector of the neural network
The dimension is a single float64-Score in [0 1] based on the length of the
text.
The value can be interpreted as likelihood of the text beeing a copyright 
protected pdf.
The value is based on the length of the extracted UTF-8-String from the pdf
using pdfminer.

@author: Mats Leon Richter
"""

from os.path import join, realpath, dirname, isdir, basename
MOD_PATH = dirname(realpath(__file__))
from doc_globals import*

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
#from pdfminer.pdfparser import PDFPage
try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO
import numpy as np
import scipy.stats as s

class TextScore:
    
    def __init__(self,txt=False):
        self.name = ["tb_score"]
        self.txt = txt
        if not txt:
            self.mean = 9109.10716436
            self.std = 22201.1272775
        else:
            #if txt files are used, the only contain a single page
            self.mean = 9109.10716436
            self.std = 22201.1272775
#            self.mean = 35.7563871896
#            self.std = 124.842470114
        if  not self.load_data():
            print("loading data for textscore failed, using default values instead...")
        self.path = 'txt_files_full'
        return
        
    # @param pages number of pages to transform to text starting from the first
    # if the argument is set to -1, all pages are read 
    # @return a utf-8 coded string from the pdf. 
    def convert_pdf_to_txt(self,fp,pages=-1):
        try:
            rsrcmgr = PDFResourceManager()
            retstr = StringIO()
            codec = 'utf-8'
            laparams = LAParams()
            device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            password = ""
            maxpages = 0
            caching = True
            pagenos=set()
            for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):
                if(pages==0):
                    break
                interpreter.process_page(page)
                pages -= 1
        
            text = retstr.getvalue()
            #fp.close()
            device.close()
            retstr.close()
            return text
        except:
            #print('troubleshooting')
            return 'ERROR'

    def get_txt(self,fp):
        path = basename(fp.name.replace('.pdf','.txt'))
        path = join(TXT_PATH, path)
        #print(path)
        fp_txt = open(path,'r')
        txt = ''
        while(True):
            try:
                tmp = fp_txt.read(1000)
                if tmp == '':
                    break
                txt += tmp
            except:
                break
        fp_txt.close()
        return txt

    def load_data(self):
        try:
            fp = open(join(MOD_PATH,'txtscore_pdf.txt'),'r')
            self.mean = float(fp.readline())
            self.std = float(fp.readline())
        except:
            return False
        return True

    #@param filepointer a pointer to a pdf file
    #@param metapointer a pointer to the metadata, this parameter is not used
    #@return float64 [0 1] probabiliy for the pdf  beeing copyright protected     
    def get_function(self,filepointer, metapointer = None):
        if(self.txt):
            try:
                x = len(self.get_txt(filepointer))
            except:
                return np.nan
        else:
            x = len(self.convert_pdf_to_txt(filepointer,-1))
        return s.norm.pdf(self.mean,self.std,x)


    def train(self,filenames,classes,metalist = None):
        """
        @param filenames:   a list of paths, leading to training files
        @param classes:     a list of classifications, in the same order as the filenames

        This function replaces the parameters of the gaussian pdf by new ones based on the files and classifications
        provided by the files
        """
        len_list = list()

        for i in range(len(filenames)):
            if(classes[i] == 'True'):
                continue
            try:
                with open(join(PDF_PATH,filenames[i]),'r') as fp:
                    if(self.txt):
                        len_list.append(len(self.get_txt(fp)))
                    else:
                        len_list.append(len(self.convert_pdf_to_txt(fp)))
            except:
                continue
            self.mean = np.mean(len_list)
            self.std = np.std(len_list)
            fp = open(join(MOD_PATH, 'txtscore_pdf.txt'),'w+')
            fp.write(str(self.mean)+'\n')
            fp.write(str(self.std)+'\n')
        print(len(len_list))
        print(self.mean)
        print(self.std)
        return