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

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO
import numpy as np
import scipy.stats as s

class TextScore:
    
    def __init__(self):
        self.mean = 9109.10716436
        self.std = 22201.1272775
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
    #@param filepointer a pointer to a pdf file
    #@param metapointer a pointer to the metadata, this parameter is not used
    #@return float64 [0 1] probabiliy for the pdf  beeing copyright protected     
    def get_function(self,filepointer, metapointer = None):
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
            with open('files/'+filenames[i],'r') as fp:
                try:
                    len_list.append(len(self.convert_pdf_to_txt(fp)))
                except:
                    continue
            self.mean = np.mean(len_list)
            self.std = np.std(len_list)

        return