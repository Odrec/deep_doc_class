#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 16:03:34 2019

@author: odrec
"""
import sys, json, csv, os
from glob import glob
from os.path import join
from collections import OrderedDict
from time import strftime

if __name__ == "__main__":
    args = sys.argv
    results_path = args[1]
    report_files = glob(join(results_path,"report*.{}".format('json')))
    indexes_sample = []
    for i,f in enumerate(report_files):
        if 'sample' in f:
            indexes_sample.append(i)
    for i in sorted(indexes_sample, reverse=True):
        del report_files[i]
    positive_classified = 0
    positive_classified_pages = []
    positive_classified_participants_pages = []
    documents_prob_08 = 0
    documents_prob_pages_08 = []
    participants_pages_prob_08 = []
    documents_prob_06 = 0
    documents_prob_pages_06 = []
    participants_pages_prob_06 = []
    documents_prob_04 = 0
    documents_prob_pages_04 = []
    participants_pages_prob_04 = []
    documents_prob_02 = 0
    documents_prob_pages_02 = []
    participants_pages_prob_02 = []
    documents_prob_under_02 = 0
    documents_prob_under_pages_02 = []
    participants_pages_prob_under_02 = []
    report_dict = OrderedDict()
    report_dict['Total documents'] = 0
    report_dict['Positively Classified'] = 0
    report_dict['Probability over 0.8'] = 0
    report_dict['Probability over 0.6'] = 0
    report_dict['Probability over 0.4'] = 0
    report_dict['Probability over 0.2'] = 0
    report_dict['Probability under 0.2'] = 0
    report_dict['Pages Positively Classified'] = 0
    report_dict['Pages Classified over 0.8'] = 0
    report_dict['Pages Classified over 0.6'] = 0
    report_dict['Pages Classified over 0.4'] = 0
    report_dict['Pages Classified over 0.2'] = 0
    report_dict['Pages Classified under 0.2'] = 0
    report_dict['Number of courses'] = len([0])
    report_dict['Pages x Participants positively classified'] = 0
    report_dict['Pages x Participants over 0.8'] = 0
    report_dict['Pages x Participants over 0.6'] = 0
    report_dict['Pages x Participants over 0.4'] = 0
    report_dict['Pages x Participants over 0.2'] = 0
    report_dict['Pages x Participants under 0.2'] = 0
    number_of_files = len(report_files)
    ts = False
    td = False
    tp = False
    print(report_files)
    for f in report_files:
        with open(f) as f_in:
            data = json.load(f_in)
            report_dict['Total documents'] += data['Total documents']
            report_dict['Positively Classified'] += data['Positively Classified']
            report_dict['Probability over 0.8'] += data['Probability over 0.8']
            report_dict['Probability over 0.6'] += data['Probability over 0.6']
            report_dict['Probability over 0.4'] += data['Probability over 0.4']
            report_dict['Probability over 0.2'] += data['Probability over 0.2']
            report_dict['Probability under 0.2'] += data['Probability under 0.2']
            report_dict['Pages Positively Classified'] += data['Pages Positively Classified']
            report_dict['Pages Classified over 0.8'] += data['Pages Classified over 0.8']
            report_dict['Pages Classified over 0.6'] += data['Pages Classified over 0.6']
            report_dict['Pages Classified over 0.4'] += data['Pages Classified over 0.4']
            report_dict['Pages Classified over 0.2'] += data['Pages Classified over 0.2']
            report_dict['Pages Classified under 0.2'] += data['Pages Classified under 0.2']
            if 'Number of courses' in data: report_dict['Number of courses'] += data['Number of courses']
            if 'Pages x Participants positively classified' in data: report_dict['Pages x Participants positively classified'] += data['Pages x Participants positively classified']
            if 'Pages x Participants over 0.8' in data: report_dict['Pages x Participants over 0.8'] += data['Pages x Participants over 0.8']
            if 'Pages x Participants over 0.6' in data: report_dict['Pages x Participants over 0.6'] += data['Pages x Participants over 0.6']
            if 'Pages x Participants over 0.4' in data: report_dict['Pages x Participants over 0.4'] += data['Pages x Participants over 0.4']
            if 'Pages x Participants over 0.2' in data: report_dict['Pages x Participants over 0.2'] += data['Pages x Participants over 0.2']
            if 'Pages x Participants under 0.2' in data: report_dict['Pages x Participants under 0.2'] += data['Pages x Participants under 0.2']
            if 'Average time preprocessing structure per file' in data: 
                report_dict['Average time preprocessing structure per file'] += data['Average time preprocessing structure per file']
                ts = True
            if 'Average time preprocessing deep features per file' in data: 
                report_dict['Average time preprocessing deep features per file'] += data['Average time preprocessing deep features per file']
                td = True
            if 'Average time predicting results per file' in data: 
                report_dict['Average time predicting results per file'] += data['Average time predicting results per file']
                tp = True
    if ts: report_dict['Average time preprocessing structure per file'] /= number_of_files
    if td: report_dict['Average time preprocessing deep features per file'] /= number_of_files
    if tp: report_dict['Average time predicting results per file'] /= number_of_files
    
    for f in report_files:
        os.remove(f)
    timestr = strftime("%Y%m%d-%H%M%S")
    report_name_json = 'report_'+timestr+'_merged.json'
    with open(join(results_path, report_name_json), 'w') as fp:
        json.dump(report_dict, fp)
    report_name_csv = 'report_'+timestr+'_merged.csv'
    with open(join(results_path,report_name_csv),'w') as fp:
        w = csv.writer(fp)
        w.writerows(report_dict.items())

    