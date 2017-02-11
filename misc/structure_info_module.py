#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 10:37:29 2017

@author: odrec
"""

import sys, csv
import numpy as np
from os.path import basename, join, splitext, isfile, isdir

class Structure_Info:
    
    def __init__(self): 
        return None

    #@return ratio of text boxes to image boxes, ratio of text boxes per page, ratio of words per text box, mean area of text boxes
    #ratio of image boxes per page, mean area of image boxes    
    def get_function(self, filename, metapointer = None):
        
        try:
            elements_list = self.get_struct_elements(filename)
            num_pages = self.get_num_pages(elements_list)
            textbox_list = self.get_textboxes(elements_list)
            imagebox_list = self.get_imageboxes(elements_list)
            ntb, txtinfo = self.get_textbox_info(textbox_list, num_pages)
            nib, imginfo = self.get_imagebox_info(imagebox_list, num_pages)
            if nib != 0:
                ratio_tb_ib = ntb/nib
            else:
                ratio_tb_ib = 0
            return ratio_tb_ib, txtinfo, imginfo
        except:
            return [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

    def get_struct_elements(self, filename):
        file_elements = []
        with open("output_text_structure.csv", "r") as f:
            reader = csv.reader(f)
            for i, r in enumerate(reader):
                if len(r) > 0:
                    if r[0] == filename:
                        file_elements.append(r)
            
        return file_elements
        
    def get_num_pages(self, elements_list):
        return max([int(e[1]) for e in elements_list])
        
    def get_textboxes(self, elements_list):
        tb_result = []
        for e in elements_list:
            if "Text" in e[2]:
                tb_result.append(e)
        return tb_result
        
    def get_textbox_info(self, textbox_list, num_pages):
        num_textboxes = len(textbox_list)
        ratio_txtbox_pages = num_textboxes/num_pages
        num_words = self.get_num_words(textbox_list)
        ratio_words_txtbox = num_words/num_textboxes
        textbox_size_avg = self.get_tb_size_avg(textbox_list)
        
        return num_textboxes, [ratio_txtbox_pages, ratio_words_txtbox, textbox_size_avg]
    
    def get_num_words(self, textbox_list):
        num_words = 0
        for e in textbox_list:
            num_words += len(e[7].split())
        return num_words
        
    def get_tb_size_avg(self, textbox_list):
        area = []
        for e in textbox_list:
            xs = float(e[5])-float(e[3])
            ys = float(e[6])-float(e[4])
            area.append(xs*ys)
        return np.mean(area)
        
    def get_imageboxes(self, elements_list):
        ib_result = []
        for e in elements_list:
            if "Image" in e[2]:
                ib_result.append(e)
        return ib_result
        
    def get_imagebox_info(self, imagebox_list, num_pages):
        num_imageboxes = len(imagebox_list)
        ratio_imgbox_pages = num_imageboxes/num_pages
        imagebox_size_avg = self.get_ib_size_avg(imagebox_list)
        
        return num_imageboxes, [ratio_imgbox_pages, imagebox_size_avg]
        
    def get_ib_size_avg(self, imagebox_list):
        area = []
        for e in imagebox_list:
            xs = float(e[5])-float(e[3])
            ys = float(e[6])-float(e[4])
            area.append(xs*ys)
        return np.mean(area)
        
    
if __name__ == "__main__":
    
    SI = Structure_Info()

    args = sys.argv
    
    filename = splitext(basename(args[1]))[0]

    print(SI.get_function(filename))
