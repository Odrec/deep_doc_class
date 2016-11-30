# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 19:58:54 2016

This module calculates the data per page for a given pdf

@author: Mats Leon Richter
"""

from os.path import join, realpath, dirname, isdir, basename
MOD_PATH = dirname(realpath(__file__))
from doc_globals import*

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
import os, numpy as np
from pdfminer.pdfpage import PDFPage
from PyPDF2 import PdfFileReader
#from pdfminer.pdfparser import PDFPage

class Page_Size_Module:

    def __init__(self, raw = True):
        self.raw = raw
        self.name = ["page_size_ratio", "pages"]
        return


    def enum_pages(self,fp):
        rsrcmgr = PDFResourceManager()
        codec = 'utf-8'
        password = ""
        maxpages = 0
        caching = True
        pagenos=set()
        length = 0
        try:
            for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):
                length += 1
        except:
            return np.nan
        return float(length)

    def get_function(self,filepointer, metapointer = None):
        """
        @input filepointer      the pointer to a pdf file
        @input metapointer      not used

        @return                 number of kilobytes per page
        """
        #get filesize in kilobytes
        size = float(os.path.getsize(filepointer.name)/(1024.0))
        
        reader = PdfFileReader(filepointer)
        pages = reader.getNumPages()

        #get number of pages
        #pages = self.enum_pages(filepointer)
        if(not self.raw):
            if pages != np.nan:
                return size/pages
            else:
                return pages
        else:
            return np.array([size,pages])

    def train(self,filenames,classes,metalist = None):
        return