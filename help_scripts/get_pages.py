#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 16:59:23 2019

@author: odrec
"""
import subprocess, sys, json, os
from multiprocessing import Pool
from os.path import basename, splitext



def process(file_path):

    output = subprocess.Popen(["pdfinfo", file_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE).communicate()[0].decode(errors='ignore')
        
        
    
    count_pages = 0
    
    if output:
        lines = output.split('\n')
        for line in lines:
            line_lower = line.lower()
            if 'pages:' in line_lower:
                cp = line_lower.split(':')[1]
                count_pages = int(cp)
            
    return count_pages

if __name__ == "__main__":
    args = sys.argv
    files = args[1]
    from os import path
    from glob import glob  
    files = glob(path.join(files,"*.{}".format('pdf')))
    print(files)
    pool = Pool(1)
    res = pool.map(process, files)
    pool.close()
    pool.join() 
    res_fix={}
    for i,x in enumerate(res):
        print(files[i])
        res_fix[splitext(basename(files[i]))[0]] = x
    ids = list(res_fix.keys())
    file = '../data/preprocessing_data/features.json'
    with open(file, "r") as jsonFile:
        json_data = json.load(jsonFile)
    os.remove(file)
    for i_d in ids:
        if 'number_pages' in json_data[i_d]: pass
        else: json_data[i_d]['number_pages'] = res_fix[i_d]
    with open(file, "w") as jsonFile:
                json.dump(json_data, jsonFile)
                
    with open(file, "r") as jsonFile:
        json_data = json.load(jsonFile)
    for d_i in ids:
        if 'number_pages' in json_data[d_i]:
            print("Number pages exists in", d_i)
        else:
            print("No number pages in", d_i)
            
    print(len(ids))

        