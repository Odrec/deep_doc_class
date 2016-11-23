__author__ = 'matsrichter'


from doc_globals import*
from os.path import join, realpath, dirname, isdir
import csv
import os
import sys

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage# if the argument is set to -1, all pages are read

from PyPDF2 import PdfFileReader, PdfFileWriter, utils

import subprocess
import json
import re

try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO

# @return a utf-8 coded string from the pdf.

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
    except:
        print('troubleshooting')
        text = ''
    return text

# def pdf_to_json_pdfminer(fp):
#         rsrcmgr = PDFResourceManager()
#         retstr = StringIO()
#         codec = 'utf-8'
#         laparams = LAParams()
#         device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
#         interpreter = PDFPageInterpreter(rsrcmgr, device)
#         password = ""
#         caching = True
#         for page in PDFPage.get_pages(fp, password=password,caching=caching, check_extractable=True):
#             interpreter.process_page(page)

#         text = retstr.getvalue()
#         device.close()
#         retstr.close()
#     return text

def pdf_to_json_gh(fp):
    output = subprocess.Popen(["gs", "-dNOPAUSE", "-dBATCH", "-sDEVICE=txtwrite", "-sOutputFile=-",fp], stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0].decode(errors='ignore')
    
    page_break_regex = "Page [0-9]+\n"
    pages = re.split(page_break_regex, output)
    if(len(pages)==1):
        print("Nothing found in %s!" % (fp,))
        return None, False
    try:
        gs_page_num = int(pages[0].split(" ")[-1][0:-2])
    except ValueError:
        print(pages[0])
        return None, False
    found_page_num = len(pages)-1
    # if(not(gs_page_num==found_page_num)):
    #     print("File: %s caused some error with the pages! gs found %d pages but %d where splitted." %  (fp, gs_page_num, found_page_num))
    #     return None
    # else:
    page_dict = {}
    empty_pages = 0
    for i in range(1, len(pages)):
        if(pages[i]==""):
            empty_pages +=1
        page_dict[i] = pages[i]
    if(empty_pages>0):
        print("Empty pages: %d" % (empty_pages,))
    else:
        print("Succes!")
    return page_dict, empty_pages>0

# def pdf_to_json_pypdf(fp):
#     json_dict = {}
#     inputpdf = PdfFileReader(fp,strict=False)
#     num_pages = inputpdf.getNumPages()
#     for i in range(0,num_pages):
#         json_dict[i] = inputpdf.getPage(i).extractText()


def extract_text(filenames, target,source,pages = 1):
    """

    :param filenames: list of pdffiles to extract pdfs from
    :param target:    targetfolder
    :param source:    source folder
    :param pages:     number of pages per file
    :return:          True if done, else False
    """
    b = TextScore()
    global c
    for f in filenames:
        try:
            c += 1
            print(str(c)+'/'+str(len(filenames)),end='\r')
            fp = open(source+f+'.pdf','rb')
            tf = open(target+f+'.txt','w+')
            txt = b.convert_pdf_to_txt(fp,pages)
            tf.write(txt)
        except:
            e = sys.exc_info()[0]
            print(e)
            fp.close()
            tf.close()
            #print("Troubleshooting")
    return True

if __name__=="__main__":

    source = join(DATA_PATH, "pdf_files")
    target = join(DATA_PATH, "json_txt_files")

    if(isdir(source)):
        filenames = os.listdir(source)
    else:
        print("source not a directory!")
        sys.exit(1)

    if(not(isdir(target))):
        os.mkdir(target)

    counter = 0
    empty_cnt = 0
    next_perc = 0.01
    for filename in filenames:
        filepath = join(source,filename)
        page_dict, empty = pdf_to_json_gh(filepath)
        if(empty):
            empty_cnt +=1
        json_path = join(target, filename[:-3]+"json")
        counter += 1
        if(counter/len(filenames)>next_perc):
            print("%.2f %% done" % (next_perc*100))
            next_perc += 0.01

        with open(json_path, 'w') as fp:
            json.dump(page_dict, fp, indent=4)

    print("Done!!!")
    print("%.2f %% had empty pages!" % (empty_cnt/len(filenames)*100))



