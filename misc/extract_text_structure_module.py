# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 14:13:00 2016

@author: odrec
"""

import sys, os
import csv, json
from multiprocessing import Pool

from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBox, LTTextLine, LTFigure, LTImage

from os.path import basename, join, splitext, isfile, isdir
from doc_globals import*

def parse_layout(filename, dict_structure, page, lt_obj, num_obj, num_subobj=0):
    
    """Function to recursively parse the layout tree."""
    
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
        for page in PDFPage.create_pages(doc):
            num_page += 1
            interpreter.process_page(page)
            layout = device.get_result()
            num_obj = 0
            for lt_obj in layout:
                parse_layout(filename, dict_structure, num_page, lt_obj, num_obj)
                num_obj += 1
    except:
        print("Failed to extract structure from file ",basename(file))
        pass

    return dict_structure

if __name__ == "__main__":

    args = sys.argv
    usage = "Usage: python extract_text_structure_module.py file|path_to_files|file_list_csv [-c number_of_cores]"
    largs = len(args)
    if largs >= 2:
        f = args[1]
        names_list = []
        files = []
        if isfile(f):
            ext = f[-3:]
            if ext == 'csv':
                with open(f, "r") as fl:
                    reader = list(csv.reader(fl, delimiter=","))
                    reader = reader[0]
                    for i, line in enumerate(reader):
                        for root, dirs, fls in os.walk(PDF_PATH):
                            for name in fls:
                                if name == line:
                                    if not name in names_list:
                                        files.append(join(root,name))
                                        names_list.append(name)
            elif ext == 'pdf':
                files.append(f)
        elif isdir(f):
            for root, dirs, fls in os.walk(f):
                for name in fls:
                    if splitext(basename(name))[1] == '.pdf':
                        files.append(join(root,name))
        else:
            print("Error: You need to specify a pdf file or path or csv file with the list of files.")
            print(usage)
            sys.exit(1)
            
        if largs == 4:
            num_cores = int(args[4])
        else:
            num_cores = 1
    
    else:
        print(usage)
        sys.exit(1)    
    
            
    print(len(files))
    
    pool = Pool(num_cores)
    res = pool.map(process_file, files)
    res_fix={}
    for x in res:    
        res_fix.update(x)
    
    
    with open("output_text_structure.json", 'w') as fp:
        json.dump(res_fix, fp)
