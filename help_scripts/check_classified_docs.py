#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 16:03:34 2019

@author: odrec
"""
import sys, json, csv
from glob import glob
from os.path import join

if __name__ == "__main__":
    args = sys.argv
    result_files = glob(join(args[1],"result*.{}".format('json')))
    processed_files_file = '../data/processed_files.csv'
    file_ids = []
    with open(processed_files_file, 'r') as f:
        reader = csv.reader(f)
        file_ids = list(reader)[0]
    for f in result_files:
        with open(f) as f_in:
            data = json.load(f_in)
        file_ids += list(data.keys())
    for i,f in enumerate(file_ids):
        file_ids[i] = file_ids[i].split("-")[0]
    file_ids = list(set(file_ids))
    with open(processed_files_file, 'w', newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(file_ids)

    