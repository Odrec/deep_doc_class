#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 10:37:29 2017
@author: odrec
"""

import sys, os
import csv, json, codecs
import numpy as np
from multiprocessing import Pool

from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBox, LTTextLine, LTFigure, LTImage

from os.path import basename, join, splitext, isfile, isdir, realpath

SRC_DIR = os.path.abspath(join(join(realpath(__file__), os.pardir),os.pardir))
sys.path.append(SRC_DIR)
sys.path.append(join(realpath(__file__), os.pardir))
from doc_globals import*

FEATURES_NAMES = ['ratio_tb_ib','ration_tb_pages','ratio_words_tb','avg_tb_size', 'ratio_words_page', 'avg_ib_size','ratio_ib_pages', 'copyright_symbol']


def get_pdf_structure(file_path, structure_path=None):
   
    doc_id = splitext(basename(file_path))[0]
    # get pdf structure dict information
    pdf_structure_data = None
    pdf_structure_dict = None

    if(not(structure_path is None) and isfile(structure_path)):
        with open(structure_path,"r") as f:
            pdf_structure_data = json.load(f)
        if(doc_id in pdf_structure_data):
            pdf_structure_dict = pdf_structure_data[doc_id]
            if()
        else:
            pdf_structure_dict, file_path = extract_pdf_structure_values(file_path)
            pdf_structure_data[doc_id] = pdf_structure_dict
            with open(structure_path,"w") as f:
                json.dump(pdf_structure_data, f, indent=4)
    else:
        pdf_structure_dict, text = extract_pdf_structure_values(file_path)

    return pdf_structure_dict,text

def extract_pdf_structure_values(file):
    struct_data, text = process_file(file)
    file_key = splitext(basename(file))[0]
    struct_data = struct_data[file_key]
    
    struct_dict = get_structure_values_dict(struct_data, text)
    return struct_dict, text

def pre_extract_pdf_structure_values(doc_dir, doc_ids=None,
    structure_file=None,
    boxinfo_file=None,
    text_dir=None,
    num_cores=1):
    
    if(boxinfo_file is None or not(isfile(boxinfo_file)) or text_dir is None or not(isdir(text_dir))):
        boxinfo, texts_dict = pre_extract_pdf_structure_boxinfo(doc_dir, doc_ids, boxinfo_file=boxinfo_file, num_cores=num_cores)
    else:
        with codecs.open(boxinfo_file, 'r', encoding='utf-8', errors='replace') as j:
            json_data=j.read()
            boxinfo = json.loads(json_data)

        texts_dict = {}
        if(doc_ids is None):
            doc_ids = []
            for root, dirs, fls in os.walk(doc_dir):
                for name in fls:
                    if splitext(basename(name))[1] == '.pdf':
                        files.append(splitext(basename(name))[0])
        for d_id in doc_ids:
            with open(join(text_dir,doc_id+"txt"), 'r') as txtfile:
                text = txtfile.readlines()
            texts_dict[d_id] = text

    pdf_structure_data = {}

    for doc_id, box_data in boxinfo.items():
        pdf_structure_data[doc_id] = get_structure_values_dict(box_data, text)

    if(not(structure_file is None)):
        with open(structure_path, 'w') as fp:
            json.dump(pdf_structure_data, fp, indent=4)

    return pdf_structure_data, texts_dict

def pre_extract_pdf_structure_boxinfo(doc_dir, doc_ids=None,
	boxinfo_file=None,
    text_dir=None,
	num_cores=1):
    files = []
    if isdir(doc_dir):
        if(doc_ids is None):
            for root, dirs, fls in os.walk(doc_dir):
                for name in fls:
                    if splitext(basename(name))[1] == '.pdf':
                        files.append(join(root,name))
        else:
            for d_id in doc_ids:
                files.append(join(doc_dir, d_id+".pdf"))
    else:
        print("Error: You need to specify a path to the folder containing all files.")
        sys.exit(1)

    pool = Pool(num_cores)
    res = pool.map(process_file, files)
    boxinfo_dict={}
    texts_dict={}
    for text,boxinfo in res:    
        boxinfo_dict.update(boxinfo)
        texts_dict.update({boxinfo.keys()[0]:text})
    
    if(not(boxinfo_file is None)):
        with open(boxinfo_file, 'w') as fp:
            json.dump(boxinfo_dict, fp, indent=4)

    return boxinfo_dict, texts_dict

##### Getting the boxinformation ######
                
def process_file(file):
    filename = splitext(basename(file))[0]
    fp = open(file, 'rb')
    dict_structure = {}
    dict_structure[filename] = {}
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
    except:
        print("Failed to extract structure from file ",basename(file))
        pass

    return (dict_structure,text)

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
                text+=lt_obj.get_text()

    return text
##### Functions for computing the values ######
    
def get_structure_values_dict(boxinfo, text):
    struct_dict = {}
    #try:
    num_pages = get_num_pages(boxinfo)
    textbox_list, imagebox_list = get_boxes(boxinfo)
    ntb, txtinfo = get_textbox_info(textbox_list, num_pages, text)
    nib, imginfo = get_imagebox_info(imagebox_list, num_pages)
    copyright_symbol = text_contains_copyright_symbol(text)
    if nib != 0:
        ratio_tb_ib = ntb/nib
    else:
        ratio_tb_ib = 0

    vals = [ratio_tb_ib]+txtinfo+imginfo+copyright_symbol

    for i in range(len(FEATURES_NAMES)):
    	struct_dict[FEATURES_NAMES[i]] = vals[i]
    return struct_dict
    # except:
    #     for i in range(len(names)):
    #     	struct_dict[names[i]] = np.nan
    #     return struct_dict

def text_contains_copyright_symbol(text):
    symbols = re.findall(r'DOI|ISBN|©|doi|isbn', text.lower())
    return int(len(symbols)>0)

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
    
def get_textbox_info(textbox_list, num_pages, text):
    num_textboxes = len(textbox_list)
    if num_pages > 0:
        ratio_txtbox_pages = num_textboxes/num_pages
    else:
        ratio_txtbox_pages = np.nan
    num_words = len(text.split())
    if num_textboxes > 0:
        ratio_words_txtbox = num_words/num_textboxes
        textbox_size_avg = get_tb_size_avg(textbox_list)
    else:
        ratio_words_txtbox = 0
        textbox_size_avg = 0

    ratio_words_page = num_words/num_pages
    
    return num_textboxes, [ratio_txtbox_pages, ratio_words_txtbox, textbox_size_avg, ratio_words_page]
    
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

if __name__ == "__main__":

    # pre_extract_pdf_structure_boxinfo(doc_dir=join(DATA_PATH,"files_test"), doc_ids=None,
    # boxinfo_file=join(DATA_PATH,"boxinfo_files_test.json"),
    # num_cores=1)

    file = "/home/kai/Workspace/deep_doc_class/deep_doc_class/data/files_test/76933924bff424b1f2bbb8ee27430f2a.pdf"
    process_file(file)