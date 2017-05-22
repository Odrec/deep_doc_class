# coding=utf-8

import sys, os, subprocess
import numpy as np
from os.path import join, realpath, dirname, isdir, basename, splitext
MOD_PATH = dirname(realpath(__file__))
SRC_DIR = os.path.abspath(join(join(realpath(__file__), os.pardir),os.pardir))
sys.path.append(SRC_DIR)
sys.path.append(join(realpath(__file__), os.pardir))

from multiprocessing import Pool

FEATURES_NAMES = ["producer","creator","pages", "file_size", "page_rot", "page_size_x", "page_size_y"]

def pdfinfo_get_pdf_properties(file_path):
    output = subprocess.Popen(["pdfinfo", file_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE).communicate()[0].decode(errors='ignore')

    prop_dict = {}
    if not output=="":
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
            if(not(key in FEATURES_NAMES)):
                continue
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
                val = int(val)>0
            prop_dict[key] = val
        
        if not 'creator' in prop_dict:
            prop_dict['creator']= None
        if not 'producer' in prop_dict:
            prop_dict['producer']= None

    if not prop_dict:
        prop_dict = {
        "file_size": np.nan,
        "page_size_x": np.nan,
        "page_size_y": np.nan,
        "producer": "passwordprotected",
        "creator": "passwordprotected",
        "page_rot": np.nan,
        "pages": np.nan}

    return (prop_dict, file_path)

def pre_extract_pdf_properties(files, properties_file=None, num_cores=1):
    pool = Pool(num_cores)
    res = pool.map(pdfinfo_get_pdf_properties, files)
    pool.close()
    pool.join() 
    res_fix={}
    for x in res:    
        res_fix[splitext(basename(x[1]))[0]] = x[0]

    return res_fix