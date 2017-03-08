# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 14:13:00 2016

@author: odrec
"""

import sys, os
import csv

from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBox, LTTextLine, LTFigure, LTImage

from os.path import basename, join, splitext, isfile, isdir
from doc_globals import*

def parse_layout(filename, page, lt_obj):
    
    """Function to recursively parse the layout tree."""
    sublist = []

    if isinstance(lt_obj, LTTextBox) or isinstance(lt_obj, LTTextLine):
        sublist.append(filename)
        sublist.append(page)
        sublist.append(lt_obj.__class__.__name__)
        sublist.append(lt_obj.bbox[0])
        sublist.append(lt_obj.bbox[1])
        sublist.append(lt_obj.bbox[2])
        sublist.append(lt_obj.bbox[3])
        sublist.append(lt_obj.get_text())

#        print(lt_obj.__class__.__name__)            
#        print(lt_obj.bbox)
#        print(lt_obj.get_text())
    elif isinstance(lt_obj, LTFigure):
        for lt_obj_int in lt_obj:
            return parse_layout(filename, page, lt_obj_int)
        return 0
#        print(lt_obj.__class__.__name__)            
#        parse_layout(filename, page, lt_obj)  # Recursive
    elif isinstance(lt_obj, LTImage):
        sublist.append(filename)
        sublist.append(page)
        sublist.append(lt_obj.__class__.__name__)
        sublist.append(lt_obj.bbox[0])
        sublist.append(lt_obj.bbox[1])
        sublist.append(lt_obj.bbox[2])
        sublist.append(lt_obj.bbox[3])
            
    return sublist


args = sys.argv
usage = "Usage: python extract_text_structure_module.py file|path_to_files|file_list_csv"
if len(args) != 2:
    print(usage)
    sys.exit(1)
    
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
        
files_structure = []
print(len(files)) 
for i in files:
    print(basename(i))
    filename = splitext(basename(i))[0]
    fp = open(i, 'rb')
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
            for lt_obj in layout:
                res = parse_layout(filename, num_page, lt_obj)
                if res != 0:
                    files_structure.append(res)
            #if num_page == 1 break #control here how many pages you want to process per file
    except:
        print("Failed to extract structure from file ",basename(i))
        pass
            
with open("output_text_structure.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(files_structure)
