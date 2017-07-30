# coding=utf-8

import sys, os, json, subprocess
from os.path import join, realpath, dirname, isdir, basename, isfile, splitext
MOD_PATH = dirname(realpath(__file__))
SRC_DIR = os.path.abspath(join(join(realpath(__file__), os.pardir),os.pardir))
if(not(SRC_DIR in sys.path)):
    sys.path.append(SRC_DIR)
FEATURE_DIR = join(SRC_DIR,"features")
if(not(FEATURE_DIR in sys.path)):
    sys.path.append(FEATURE_DIR)

from doc_globals import*

from multiprocessing import Pool

FEATURES_NAMES = ["producer","creator","pages", "file_size", "page_rot", "page_size_x", "page_size_y"]

def pdfinfo_get_pdf_properties(file_path):
    output = subprocess.Popen(["pdfinfo", file_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE).communicate()[0].decode(errors='ignore')

    if(output==""):
        pdfinfo_dict = {
            "optimized": "passwordprotected",
            "file_size": 0.0,
            "form": "passwordprotected",
            "tagged": "passwordprotected",
            "encrypted": 0,
            "pdf_version": "passwordprotected",
            "suspects": "passwordprotected",
            "page_size_x": 0.0,
            "creationdate": "passwordprotected",
            "userproperties": "passwordprotected",
            "page_size_y": 0.0,
            "author": "passwordprotected",
            "javascript": "passwordprotected",
            "producer": "passwordprotected",
            "moddate": "passwordprotected",
            "creator": "passwordprotected",
            "page_rot": 0,
            "title": "passwordprotected",
            "pages": 0}
        return pdfinfo_dict, file_path
    prop_dict = {}
    lines = output.split('\n')[:-1]
    new_lines=[]
    for l, line in enumerate(lines):
        if ':' in line:
            new_lines.append(line)
        else:
            new_lines[-1] += ' '+line

    for line in new_lines:
        key, val = line.split(':',1)
        key = key.lower().replace(" ", "_")
        try:
            val = val.split(None,0)[0]
        except:
            prop_dict[key] = None
            continue
        if(key == "page_size"):
            val = val.split()
            prop_dict["page_size_x"] = float(val[0])
            key = "page_size_y"
            val = float(val[2])
        elif(key == "pages"):
            val = int(val)
        elif(key == "file_size"):
            val = val.split()[0]
            val = float(val)/1000
        elif(key == "page_rot"):
            val = int(int(val)>0)
        elif(key == "encrypted"):
            val = not(val=="no")
        prop_dict[key] = val

    if not 'author' in prop_dict:
        prop_dict['author']= "None"
    if not 'creator' in prop_dict:
        prop_dict['creator']= "None"
    if not 'producer' in prop_dict:
        prop_dict['producer']= "None"
    if not 'title' in prop_dict:
        prop_dict['title']= "None"

    return (prop_dict, file_path)

def get_pdf_properties(file_path, properties_path=None):

    doc_id = splitext(basename(file_path))[0]

    # get pdfinfo dict information
    pdfinfo_data = None
    pdfinfo_dict = None

    if(not(properties_path is None) and isfile(properties_path)):
        with open(properties_path,"r") as f:
            pdfinfo_data = json.load(f)
        if(doc_id in pdfinfo_data):
            pdfinfo_dict = pdfinfo_data[doc_id]
        else:
            pdfinfo_dict, filepath = pdfinfo_get_pdf_properties(file_path)
            pdfinfo_data[doc_id] = pdfinfo_dict
            with open(properties_path,"w") as f:
                json.dump(pdfinfo_data, f, indent=4)
    else:
        pdfinfo_data = {}
        pdfinfo_dict, file_path = pdfinfo_get_pdf_properties(file_path)

    if(pdfinfo_dict is None):
        pdfinfo_dict = {
            "optimized": "passwordprotected",
            "file_size": None,
            "form": "passwordprotected",
            "tagged": "passwordprotected",
            "encrypted": None,
            "pdf_version": "passwordprotected",
            "suspects": "passwordprotected",
            "page_size_x": None,
            "creationdate": "passwordprotected",
            "userproperties": "passwordprotected",
            "page_size_y": None,
            "author": None,
            "javascript": "passwordprotected",
            "producer": "passwordprotected",
            "moddate": "passwordprotected",
            "creator": "passwordprotected",
            "page_rot": None,
            "title": None,
            "pages": None}

    return pdfinfo_dict

def pre_extract_pdf_properties(doc_dir, doc_ids=None, properties_file=None, num_cores=1):
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
    res = pool.map(pdfinfo_get_pdf_properties, files)
    res_fix={}
    for x in res:
        res_fix[splitext(basename(x[1]))[0]] = x[0]

    if(not(properties_file) is None):
        with open(properties_file, 'w') as fp:
            json.dump(res_fix, fp)

    return res_fix

def load_single_property(doc_ids, properties_path, field):

    # get pdfinfo dict information
    pdfinfo_data = None
    properties = []

    with open(properties_path,"r") as f:
        pdfinfo_data = json.load(f)
        for doc_id in doc_ids:
            pdfinfo_dict = pdfinfo_data[doc_id]

            datafield = ""
            if(pdfinfo_dict==None):
                datafield = "passwordprotected"
            elif(pdfinfo_dict[field]==None):
                datafield = "None"
            else:
                datafield = pdfinfo_dict[field]
            properties.append(datafield)

    return properties

if __name__ == "__main__":

    doc_dir="../../data/pdf_files"
    prop_file="../../data/pre_extracted_data/pdf_properties_new.json"

    pre_extract_pdf_properties(doc_dir, doc_ids=None, properties_file=prop_file, num_cores=4)
