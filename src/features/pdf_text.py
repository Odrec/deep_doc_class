# coding=utf-8

import sys, os
from os.path import join, realpath, dirname, isdir, basename, isfile, splitext
MOD_PATH = dirname(realpath(__file__))
SRC_DIR = os.path.abspath(join(join(realpath(__file__), os.pardir),os.pardir))
sys.path.append(SRC_DIR)
sys.path.append(join(realpath(__file__), os.pardir))
from doc_globals import*

import csv, re, json

import subprocess
from functools import partial

from multiprocessing import Pool


def ghostcript_get_pdftext(file_path, txt_path=None, first_page=-1, last_page=-1):

    args = ["gs", "-dNOPAUSE", "-dBATCH", "-sDEVICE=txtwrite", "-sOutputFile=-",file_path]
    if(not(first_page==-1)):
        args.append("-dFirstPage=%d"%(first_page,))
    if(not(last_page==-1)):
        args.append("-dLastPage=%d"%(last_page,))
    output = subprocess.Popen(args, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE).communicate()[0].decode(errors='ignore')

    sub_regex = r'Substituting (.)+\n'
    loading_regex = r'Loading (.)+\n'
    query_regex = r'Querying (.)+\n'
    cant_find_regex = r'Can\'t find \(or can\'t open\) (.)+\n'
    didnt_find_regex = r'Didn\'t find (.)+\n'
    invalid_file_regex = r'Error: /invalidfileaccess in pdf_process_Encrypt'

    if(re.search(invalid_file_regex, output)):
        with open(out, 'w') as fp:
            json.dump(None, fp, indent=4)
        return None, 0

    output = re.sub(sub_regex, '', output)
    output = re.sub(loading_regex, '', output)
    output = re.sub(query_regex, '', output)
    output = re.sub(cant_find_regex, '', output)
    output = re.sub(didnt_find_regex, '', output)

    page_break_regex = "Page [0-9]+\n"
    pages = re.split(page_break_regex, output)

    # txt = ""
    page_dict = {}
    empty_pages = 0
    for i in range(1, len(pages)):
        if(pages[i]==""):
            empty_pages +=1
        else:
            pages[i] = pages[i].split('done.\n')[-1]
        page_dict[str(i)] = pages[i]
        # txt += pages[i]

    if(not(txt_path is None) and isdir(txt_path)):
        out = join(txt_path,basename(file_path)[:-4]+".json")
        with open(out, 'w') as fp:
            json.dump(page_dict, fp, indent=4)

    # empty_pages = float(empty_pages)/(len(pages))
    return (page_dict,file_path)

def get_pdf_texts(doc_ids,pdf_path,txt_dir):
    # if its only a single doc_id make it a list of one
    if(type(doc_ids)==str):
        doc_ids = [doc_ids]

    texts = []
    # the text is stored in a seperate json per pdf
    for d_id in doc_ids:
        json_path = join(txt_dir,d_id+".json")
        if(isfile(json_path)):
            # data in the json is a dict{page:string}
            data = json.load(open(json_path,"r"))
        else:
            filepath = pdf_path+d_id+".pdf"
            data = ghostcript_get_pdftext(filepath, txt_dir, first_page=-1, last_page=-1, save_json=True)[1]

        # if the dict is None the document was password protected
        if(data is None):
            text_data = "password_protected"
        elif(len(data)==0):
            text_data = "None"
        else:
            l_page = max(list(map(int, data.keys())))
            text_data = ""
            # concatenate the text of maximal the specified num_pages
            for i in range(1,max(l_page,num_pages)+1):
                text_data += (data[str(i)] + " ")

        texts.append(text_data)
            # # since the text can be very large give a Warning if it gets really high
            # if(sys.getsizeof(texts)>1000000000):
            #     print("Warning!!! Input data is larger than 1GB!")
    # if it was just one doc_id return a string only
    if(len(texts)==1):
        texts = texts[0]
    return texts

def get_pdf_text_features(file_path,text_path=None):

    doc_id = splitext(basename(file_path))[0]

    page_dict = {}
    if(not(text_path is None)):
        txt_file = join(text_path,doc_id+".json")
        if(isfile(txt_file)):
            with open(txt_file,"r") as f:
                page_dict = json.load(f)
        else:
            page_dict, file_path = ghostcript_get_pdftext(file_path,text_path, first_page=-1, last_page=-1)
    else:
        page_dict, file_path = ghostcript_get_pdftext(file_path,text_path, first_page=-1, last_page=-1)

    text_dict={}
    if(page_dict is None or bool(page_dict) is False):
        text_dict["text"] = "password_protected"
        text_dict["word_count"] = None
        text_dict["copyright_symbol"] = None
        return text_dict
    else:
        text = ""
        l_page = max(list(map(int, page_dict.keys())))

        # concatenate the text
        for i in range(1,l_page+1):
            page_text = page_dict[str(i)]
            text += page_text

        text_dict["text"] = text
        text_dict["word_count"] = float(len(text))/l_page
        symbols = re.findall(r'DOI|ISBN|©|doi|isbn', text)
        text_dict["copyright_symbol"] = len(symbols)>0 
        return text_dict 

def pre_extract_pdf_texts(doc_dir, txt_dir, doc_ids=None, num_cores=1):
    files = []
    if isdir(doc_dir):
        if(doc_ids is None):
            for root, dirs, fls in os.walk(doc_dir):
                for name in fls:
                    if splitext(basename(name))[1] == '.pdf':
                        files.append(join(root,name))
        else:
            for d_id in doc_ids:
                files.append(join(doc_dir,doc_ids+".pdf"))
    else:
        print("Error: You need to specify a path to the folder containing all files.")
        sys.exit(1)

    pool = Pool(num_cores)
    res = pool.map(partial(ghostcript_get_pdftext,txt_dir=txt_dir), files)
