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

FEATURES_NAMES = ["text", "word_count", "copyright_symbol"]

def ghostcript_get_pdftext(file_path, text_dir=None, first_page=-1, last_page=-1, json_format=False):

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


    page_dict = None
    text = "password_protected"
    if(not(re.search(invalid_file_regex, output))):
        output = re.sub(sub_regex, '', output)
        output = re.sub(loading_regex, '', output)
        output = re.sub(query_regex, '', output)
        output = re.sub(cant_find_regex, '', output)
        output = re.sub(didnt_find_regex, '', output)

        page_break_regex = "Page [0-9]+\n"
        pages = re.split(page_break_regex, output)

        text = str(len(pages)-1)+"\n"
        page_dict = {}
        for i in range(1, len(pages)):
            if(not(pages[i]=="")):
                pages[i] = pages[i].split('done.\n')[-1]
            page_dict[str(i)] = pages[i]
            text += pages[i]

    if(not(text_dir is None) and isdir(text_dir)):
        if(json_format):
            out = join(text_dir,basename(file_path)[:-4]+".json")
            with open(out, 'w') as fp:
                json.dump(page_dict, fp, indent=4)
        else:
            out = join(text_dir,basename(file_path)[:-4]+".txt")
            with open(out, 'w') as fp:
                fp.write(text)

    if(json_format):
        return (page_dict,file_path)
    else:
        return (text,file_path)

def get_pdf_texts_json(doc_ids,pdf_path,txt_dir=None):
    # if its only a single doc_id make it a list of one
    if(type(doc_ids)==str):
        doc_ids = [doc_ids]

    texts = []
    # the text is stored in a seperate json per pdf
    for d_id in doc_ids:
        if(not(txt_dir) is None):
            json_path = join(txt_dir,d_id+".json")
            if(isfile(json_path)):
                # data in the json is a dict{page:string}
                with open(json_path, "r") as json_file:
                    data = json.load(json_file)
            else:
                filepath = join(pdf_path,d_id+".pdf")
                data = ghostcript_get_pdftext(filepath, txt_dir, first_page=-1, last_page=-1, json_format=True)[0]
        else:
            filepath = join(pdf_path,d_id+".pdf")
            data = ghostcript_get_pdftext(filepath, txt_dir, first_page=-1, last_page=-1, json_format=True)[0]


        # if the dict is None the document was password protected
        if(data is None):
            text_data = "password_protected"
        elif(len(data)==0):
            print(data)
            text_data = "None"
        else:
            text_data, l_page = load_text_from_json(data)
        texts.append(text_data)

    # if it was just one doc_id return a string only
    if(len(texts)==1):
        texts = texts[0]
    return texts

def get_pdf_texts_txt(doc_ids,pdf_path,txt_dir=None):
    # if its only a single doc_id make it a list of one
    if(type(doc_ids)==str):
        doc_ids = [doc_ids]

    texts = []
    # the text is stored in a seperate json per pdf
    for d_id in doc_ids:
        if(not(txt_dir) is None):
            txt_path = join(txt_dir,d_id+".txt")
            if(isfile(txt_path)):
                # data in the json is a dict{page:string}
                with open(txt_path, "r") as txt_file:
                    pages = txt_file.readline()
                    text_data = txt_file.read()
            else:
                filepath = join(pdf_path,d_id+".pdf")
                text_data = ghostcript_get_pdftext(filepath, txt_dir, first_page=-1, last_page=-1, json_format=False)[0]
        else:
            filepath = join(pdf_path,d_id+".pdf")
            text_data = ghostcript_get_pdftext(filepath, txt_dir, first_page=-1, last_page=-1, json_format=False)[0]

        texts.append(text_data)

    # if it was just one doc_id return a string only
    if(len(texts)==1):
        texts = texts[0]
    return texts

def get_pdf_text_features_json(file_path,text_dir=None):

    doc_id = splitext(basename(file_path))[0]

    page_dict = {}
    if(not(text_dir is None)):
        txt_file = join(text_dir,doc_id+".json")
        if(isfile(txt_file)):
            with open(txt_file,"r") as f:
                page_dict = json.load(f)
        else:
            page_dict, file_path = ghostcript_get_pdftext(file_path,text_dir, first_page=-1, last_page=-1, json_format=True)
    else:
        page_dict, file_path = ghostcript_get_pdftext(file_path,text_dir, first_page=-1, last_page=-1, json_format=True)

    text_dict={}
    if(page_dict is None or bool(page_dict) is False):
        text_dict["text"] = "password_protected"
        text_dict["word_count"] = None
        text_dict["copyright_symbol"] = None
        return text_dict
    else:
        text, l_page = load_text_from_json(page_dict)
        text_dict["text"] = text
        text_dict["word_count"] = float(len(text))/l_page
        symbols = re.findall(r'DOI|ISBN|©|doi|isbn', text)
        text_dict["copyright_symbol"] = len(symbols)>0 
        return text_dict

def get_pdf_text_features_txt(file_path,text_dir=None):

    doc_id = splitext(basename(file_path))[0]

    text = ""
    if(not(text_dir is None)):
        txt_file = join(text_dir,doc_id+".txt")
        if(isfile(txt_file)):
            with open(txt_file,"r") as f:
                text = f.read()
        else:
            text, file_path = ghostcript_get_pdftext(file_path,text_dir, first_page=-1, last_page=-1)
    else:
        text, file_path = ghostcript_get_pdftext(file_path,text_dir, first_page=-1, last_page=-1)

    pages = int(text.split("\n")[0])

    text_dict = {}
    text_dict["text"] = text
    text_dict["word_count"] = float(len(text))/pages
    symbols = re.findall(r'DOI|ISBN|©|doi|isbn', text)
    text_dict["copyright_symbol"] = len(symbols)>0 
    return text_dict 

def pre_extract_pdf_texts(doc_dir, txt_dir, doc_ids=None, json_format=False, num_cores=1):
    files = []
    if isdir(doc_dir):
        if(doc_ids is None):
            for root, dirs, fls in os.walk(doc_dir):
                for name in fls:
                    if splitext(basename(name))[1] == '.pdf':
                        files.append(join(root,name))
        else:
            for d_id in doc_ids:
                files.append(join(doc_dir,d_id+".pdf"))
    else:
        print("Error: You need to specify a path to the folder containing all files.")
        sys.exit(1)

    pool = Pool(num_cores)

    res = pool.map(partial(ghostcript_get_pdftext_json,txt_dir=txt_dir, json_format=json), files)

def load_text_from_json(page_dict):
    text = ""
    l_page = max(list(map(int, page_dict.keys())))

    # concatenate the text
    for i in range(1,l_page+1):
        page_text = page_dict[str(i)]
        text += page_text

    return text, l_page

if __name__ == "__main__":
    doc_dir = "/home/kai/Workspace/deep_doc_class/deep_doc_class/data/files_test"
    txt_dir = "/home/kai/Workspace/deep_doc_class/deep_doc_class/data/files_test_txt"

    if(not(isdir(txt_dir))):
        os.makedirs(txt_dir)

    doc_ids = None
    doc_ids = ['776ef484a6a0508f0eddfc3556a89f22', '76ee16edcc818768a2e694146200c62e', '76ae7c120910a7830a1c0e0262d8cc5e', '767f05ad09c467135701dbb91432ee2b', '763e52fe7ed11a7384667aedc6826532', '76ee11104f97d05592728a76819886af', '770cade75e2423d0e45bdcf3bad78d23', '760ca30d4bf5a608ea1e53c1891d1135', '767d64edf6a4efd636178f75e3284984', '76884a59077eb05c5f3de8ba77f1078a', '76d424cc5ddc1e2fa73cf461d6f5234f', '773ff7184bb6aafd55c8a44c92ed2d60', '76b43039b7577b89dcad89621d42c7d5', '76933924bff424b1f2bbb8ee27430f2a', '76bdd440a7affb8f339f2b0236a48f3d', '760864e425b1310212fbe2907b8a434c', '7619c8d1ecc2044f3149bae08a57ff06', '76cd9292aea2fb04133ec7c21b9c670d', '775bd074af6686b74906ce0bff43c27a', '76761974a8b4bbaea9265080f124cd67']

    files = []
    if isdir(doc_dir):
        if(doc_ids is None):
            for root, dirs, fls in os.walk(doc_dir):
                for name in fls:
                    if splitext(basename(name))[1] == '.pdf':
                        files.append(join(root,name))
        else:
            for d_id in doc_ids:
                files.append(join(doc_dir,d_id+".pdf"))

    # for f in files:
    #     val_dict = get_pdf_text_features_txt(f)
    #     print(len(val_dict["text"]))
    #     print(val_dict["word_count"])
    #     print(val_dict["copyright_symbol"])

    texts = get_pdf_texts_json(doc_ids,doc_dir,txt_dir)
    for t in texts:
        print(len(t))

