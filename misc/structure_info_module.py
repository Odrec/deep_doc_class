#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 10:37:29 2017

@author: odrec
"""

import sys, os, json, codecs
import numpy as np
from os.path import basename, join, splitext, isfile, isdir
import extract_text_structure_module as es

class Structure_Info:
    
    def __init__(self, path=None): 
        
        if path != None:
            file = path
        else:
            file = 'output_text_structure.json'
            
        self.struct_elements = self.get_struct_elements(file)
        return None

    #@return ratio of text boxes to image boxes, ratio of text boxes per page, ratio of words per text box, mean area of text boxes
    #ratio of image boxes per page, mean area of image boxes    
    def get_function(self, file, metapointer = None):
        
        file_key = splitext(basename(file))[0]
        if file_key in self.struct_elements:
            struct_data = self.struct_elements[file_key]
        else:
            struct_data = es.process_file(file)
            struct_data = struct_data[file_key]
            
        names = ['ratio_text_image', 'ration_text_pages', 'ratio_words_box', 'avg_text_size', 'ratio_image_pages', 'avg_image_size']
        try:
            num_pages = self.get_num_pages(struct_data)
            textbox_list, imagebox_list = self.get_boxes(struct_data)
            ntb, txtinfo = self.get_textbox_info(textbox_list, num_pages)
            nib, imginfo = self.get_imagebox_info(imagebox_list, num_pages)
            if nib != 0:
                ratio_tb_ib = ntb/nib
            else:
                ratio_tb_ib = 0
            
            return names, [ratio_tb_ib]+txtinfo+imginfo
        except:
            return names, [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

    def get_json_results(self, file, results_dict):
        names, results_list = self.get_function(file)
        file_name = splitext(basename(file))[0]
        results_dict[file_name] = {}
        for i, n in enumerate(names):
            results_dict[file_name][names[i]] = results_list[i]
        return names, results_list

    def get_struct_elements(self, filename):
        file_elements = {}
        if isfile(filename):
            with codecs.open(filename, 'r', encoding='utf-8', errors='replace') as j:
                json_data=j.read()
                file_elements = json.loads(json_data)
            
        return file_elements
        
    def get_num_pages(self, struct_data):
        return len(struct_data)
        
    def get_boxes(self, struct_data):
        tb_result = []
        ib_result = []
        for k in struct_data:
            for kk in struct_data[k]:
                if "Text" in kk:
                    tb_result.append(struct_data[k][kk])
                elif "Image" in kk:
                    ib_result.append(struct_data[k][kk])
        return tb_result, ib_result
        
    def get_textbox_info(self, textbox_list, num_pages):
        num_textboxes = len(textbox_list)
        if num_pages > 0:
            ratio_txtbox_pages = num_textboxes/num_pages
        else:
            ratio_txtbox_pages = np.nan
        num_words = self.get_num_words(textbox_list)
        if num_textboxes > 0:
            ratio_words_txtbox = num_words/num_textboxes
        else:
            ratio_words_txtbox = np.nan
        textbox_size_avg = self.get_tb_size_avg(textbox_list)
        
        return num_textboxes, [ratio_txtbox_pages, ratio_words_txtbox, textbox_size_avg]
    
    def get_num_words(self, textbox_list):
        num_words = 0
        for e in textbox_list:
            num_words += len(e['text'].split())
        return num_words
        
    def get_tb_size_avg(self, textbox_list):
        area = []
        for e in textbox_list:
            xs = float(e['box3'])-float(e['box1'])
            ys = float(e['box4'])-float(e['box2'])
            area.append(xs*ys)
        return np.mean(area)
        
    def get_imagebox_info(self, imagebox_list, num_pages):
        num_imageboxes = len(imagebox_list)
        if num_pages > 0:
            ratio_imgbox_pages = num_imageboxes/num_pages
        else:
            ratio_imgbox_pages = np.nan
        imagebox_size_avg = self.get_ib_size_avg(imagebox_list)
        
        return num_imageboxes, [ratio_imgbox_pages, imagebox_size_avg]
        
    def get_ib_size_avg(self, imagebox_list):
        area = []
        for e in imagebox_list:
            xs = float(e['box3'])-float(e['box1'])
            ys = float(e['box4'])-float(e['box2'])
            area.append(xs*ys)
        return np.mean(area)
        
    
if __name__ == "__main__":
    
    args = sys.argv
    
    usage = "python structure_info_module.py [-s] [-j json_file] -f path_pdf_file(s)"
       
    save = False     
    if '-s' in args:
        save = True
    if '-j' in args:
        index_j = args.index('-j')
        if isfile(args(index_j+1)):
            SI = Structure_Info(args(index_j+1))
        else:
            print(usage)
            sys.exit(1)
    else:
        SI = Structure_Info()
    if '-f' in args:
        index_f = args.index('-f')
        path_pdf = args[index_f+1]
        files = []
        if isdir(path_pdf):
            for root, dirs, fls in os.walk(path_pdf):
                for name in fls:
                    if splitext(basename(name))[1] == '.pdf':
                        files.append(join(root,name))
        elif isfile(path_pdf):
            files = [path_pdf]
        else:
            print(usage)
            sys.exit(1)  

    results_list = []
    results_dict = {}
    for f in files:
        if save: 
            names, res_list = SI.get_json_results(f, results_dict)
            with open("output_structure_features.json", 'w') as fp:
                json.dump(results_dict, fp)
        else:
            names, res_list = SI.get_function(f)
        results_list.append(res_list)
        
    print(names)
    print(results_list)
        
    
