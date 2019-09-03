#!/usr/bin/env python3
# coding=utf-8

import subprocess
import numpy as np
from os.path import basename, splitext
import logging.config

logging.config.fileConfig(fname='log.conf', disable_existing_loggers=False)
# Get the logger specified in the file
logger = logging.getLogger("copydocLogger")
debuglogger = logging.getLogger("debugLogger")

def pdfinfo_get_pdf_properties(file_path):
    sprocess = subprocess.Popen(["pdfinfo", file_path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = sprocess.communicate()[0].decode(errors='ignore')
    
    debuglogger.debug("Processing properties for file %s",basename(file_path))
    
    #Default values
    prop_dict = {
    "file_size": -1,
    "page_size_x": -1,
    "page_size_y": -1,
    "producer": "empty",
    "creator": "empty",
    "page_rot": -1
    }

    if output:
        lines = output.split('\n')
        for line in lines:
            line_lower = line.lower()
            if ':' in line_lower:
                if 'creator' in line_lower:
                    prop_dict['creator'] = line_lower.split(':')[1]
                elif 'producer' in line_lower:
                    prop_dict['producer'] = line_lower.split(':')[1]
                elif 'file size' in line_lower:
                    val = line_lower.split(':')[1]
                    #take the first part of the value after the : and normalize
                    val = val.split()[0]
                    prop_dict['file_size'] = val
                elif 'page rot' in line_lower:
                    prop_dict['page_rot'] = int(line_lower.split(':')[1])
                elif 'page size' in line_lower:
                    val = line_lower.split(':')[1]
                    if 'x' in val:
                        val = val.split()
                        index_x = val.index('x')
                        prop_dict['page_size_x'] = float(val[index_x-1])
                        prop_dict['page_size_y'] = float(val[index_x+1])
                    else:
                        prop_dict['page_size_x'] = -1
                        prop_dict['page_size_y'] = -1
    else:
        #if there's no output
        debuglogger.error("There was no pdfinfo output for file %s.",basename(file_path))
        prop_dict = {
        "file_size": np.nan,
        "page_size_x": np.nan,
        "page_size_y": np.nan,
        "producer": "passwordprotected",
        "creator": "passwordprotected",
        "page_rot": np.nan
        }
    sprocess.terminate()
    
    return (prop_dict, file_path)

def pre_extract_pdf_properties(files, pool):
    res = pool.map(pdfinfo_get_pdf_properties, files)
    res_fix={}
    for x in res:
        res_fix[splitext(basename(x[1]))[0]] = x[0]
    return res_fix
