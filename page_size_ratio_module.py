# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 19:58:54 2016

This module calculates the data per page for a given pdf

@author: Mats Leon Richter
"""


from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
import os
from pdfminer.pdfpage import PDFPage

class Page_Size_Module:

    def __init__(self):
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
            return 1.0
        return float(length)

    def get_function(self,filepointer, metapointer = None):
        """
        @input filepointer      the pointer to a pdf file
        @input metapointer      not used

        @return                 number of kilobytes per page
        """

        #get filesize in kilobytes
        size = float(os.path.getsize(filepointer.name)/(1024.0))

        #get number of pages
        pages = self.enum_pages(filepointer)

        return size/pages

    def train(self,filenames,classes,metalist = None):
        return