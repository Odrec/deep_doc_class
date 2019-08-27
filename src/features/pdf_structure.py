#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 10:37:29 2017
@author: odrec
"""

import sys, os
import json, codecs
import numpy as np
from multiprocessing import Pool

from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBox, LTTextLine, LTFigure, LTImage

from os.path import basename, join, splitext, isfile, realpath

SRC_DIR = os.path.abspath(join(join(realpath(__file__), os.pardir),os.pardir))
sys.path.append(SRC_DIR)
sys.path.append(join(realpath(__file__), os.pardir))

FEATURES_NAMES = ['ratio_tb_ib','ratio_tb_pages','ratio_words_tb','avg_tb_size','ratio_ib_pages','avg_ib_size']


def pre_extract_pdf_structure_values(files, structure_file=None, boxinfo_file=None, num_cores=1):
    
    if(boxinfo_file is None or not(isfile(boxinfo_file))):
        text, boxinfo = pre_extract_pdf_structure_boxinfo(files, boxinfo_file=boxinfo_file, num_cores=num_cores)
    else:
        with codecs.open(boxinfo_file, 'r', encoding='utf-8', errors='replace') as j:
            json_data=j.read()
            boxinfo = json.loads(json_data)

    pdf_structure_data = {}

    for doc_id, box_data in boxinfo.items():
        pdf_structure_data[doc_id] = get_structure_values_dict(box_data)

    return text, pdf_structure_data

def pre_extract_pdf_structure_boxinfo(files, boxinfo_file=None, num_cores=1):
    pool = Pool(num_cores)
    res = pool.map(process_file, files)
    pool.close()
    pool.join() 
    res_text = {}
    res_dict = {}
    for r in res:
        res_text.update(r[0])
        res_dict.update(r[1])

    return res_text, res_dict

##### Getting the boxinformation ######
def process_file(file):
    filename = splitext(basename(file))[0]
    fp = open(file, 'rb')
    dict_structure = {}
    dict_structure[filename] = {}
    text_dict = {}
    text_dict[filename] = {}
    try:
        parser = PDFParser(fp)
        doc = PDFDocument(parser)
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        num_page = 0
        text = ""
        for page in PDFPage.create_pages(doc):
            num_page += 1
            interpreter.process_page(page)
            layout = device.get_result()
            num_obj = 0
            for lt_obj in layout:
                text += parse_layout(filename, dict_structure, num_page, lt_obj, num_obj)
                num_obj += 1      
        text_dict[filename]["text"] = text
    except:
        print("Failed to extract structure from file ",basename(file))
        text_dict[filename]["text"] = " "
        pass
    
    return (text_dict, dict_structure)

def parse_layout(filename, dict_structure, page, lt_obj, num_obj, num_subobj=0):
    
    """Function to recursively parse the layout tree."""
    text = ""
    
    if isinstance(lt_obj, LTFigure):
        num_subobj = 0
        for lt_obj_int in lt_obj:
            parse_layout(filename, dict_structure, page, lt_obj_int, num_obj, num_subobj)
            num_subobj += 1
    else:
        if isinstance(lt_obj, LTImage) or isinstance(lt_obj, LTTextBox) or isinstance(lt_obj, LTTextLine):
            if not page in dict_structure[filename]:
                dict_structure[filename][page] = {}
            dict_structure[filename][page][lt_obj.__class__.__name__+'-'+str(num_obj)+'-'+str(num_subobj)] = {}
            dict_structure[filename][page][lt_obj.__class__.__name__+'-'+str(num_obj)+'-'+str(num_subobj)].update({'box1': lt_obj.bbox[0],
            'box2': lt_obj.bbox[1],'box3': lt_obj.bbox[2],'box4': lt_obj.bbox[3]})
                
            if not isinstance(lt_obj, LTImage):
                dict_structure[filename][page][lt_obj.__class__.__name__+'-'+str(num_obj)+'-'+str(num_subobj)].update({'text':lt_obj.get_text()})
                text += lt_obj.get_text()
                
    return text

##### Functions for computing the values ######
    
def get_structure_values_dict(boxinfo):
    struct_dict = {}
    num_pages = get_num_pages(boxinfo)
    textbox_list, imagebox_list = get_boxes(boxinfo)
    ntb, txtinfo = get_textbox_info(textbox_list, num_pages)
    nib, imginfo = get_imagebox_info(imagebox_list, num_pages)
    if nib != 0:
        ratio_tb_ib = ntb/nib
    else:
        ratio_tb_ib = 0

    vals = [ratio_tb_ib]+txtinfo+imginfo

    for i in range(len(FEATURES_NAMES)):
        struct_dict[FEATURES_NAMES[i]] = vals[i]

    return struct_dict

def get_num_pages(struct_data):
    return len(struct_data)
    
def get_boxes(struct_data):
    tb_result = []
    ib_result = []
    for k in struct_data:
        for kk in struct_data[k]:
            if "Text" in kk:
                tb_result.append(struct_data[k][kk])
            elif "Image" in kk:
                ib_result.append(struct_data[k][kk])
                
    return tb_result, ib_result
    
def get_textbox_info(textbox_list, num_pages):
    num_textboxes = len(textbox_list)
    if num_pages > 0:
        ratio_txtbox_pages = num_textboxes/num_pages
    else:
        ratio_txtbox_pages = np.nan
    num_words = get_num_words(textbox_list)
    if num_textboxes > 0:
        ratio_words_txtbox = num_words/num_textboxes
        textbox_size_avg = get_tb_size_avg(textbox_list)
    else:
        ratio_words_txtbox = 0
        textbox_size_avg = 0
            
    return num_textboxes, [ratio_txtbox_pages, ratio_words_txtbox, textbox_size_avg]

def get_num_words(textbox_list):
    num_words = 0
    for e in textbox_list:
        text = e.pop('text', None)
        if not text is None:
            num_words += len(text.split())
    return num_words
    
def get_tb_size_avg(textbox_list):
    area = []
    for e in textbox_list:
        xs = float(e['box3'])-float(e['box1'])
        ys = float(e['box4'])-float(e['box2'])
        area.append(xs*ys)
        
    return np.mean(area)
    
def get_imagebox_info(imagebox_list, num_pages):
    num_imageboxes = len(imagebox_list)
    if num_pages > 0:
        ratio_imgbox_pages = num_imageboxes/num_pages
    else:
        ratio_imgbox_pages = np.nan
    if(len(imagebox_list)>0):
        imagebox_size_avg = get_ib_size_avg(imagebox_list)
    else:
        imagebox_size_avg = 0
    
    return num_imageboxes, [ratio_imgbox_pages, imagebox_size_avg]
    
def get_ib_size_avg(imagebox_list):
    area = []
    for e in imagebox_list:
        xs = float(e['box3'])-float(e['box1'])
        ys = float(e['box4'])-float(e['box2'])
        area.append(xs*ys)
        
    return np.mean(area)