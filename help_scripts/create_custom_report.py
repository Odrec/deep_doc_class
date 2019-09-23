#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 16:59:23 2019

@author: odrec
"""
import subprocess, sys, json, csv
from glob import glob  
from os.path import join
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

def write_report(results_path, final_prediction, batch_ids, count_pages, courses, number_participants, pages_limit):
    print(len(courses))
    if courses: courses = list(dict.fromkeys(courses))
    print(len(final_prediction),len(batch_ids),len(count_pages),len(courses),len(number_participants))
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
    for i,v in enumerate(final_prediction):
        if final_prediction[i] > 0.2:
            documents_prob_02 += 1
            documents_prob_pages_02.append(count_pages[i])
            if number_participants: participants_pages_prob_02.append(number_participants[i]*count_pages[i])
            if final_prediction[i] > 0.4:
                documents_prob_04 += 1
                documents_prob_pages_04.append(count_pages[i])
                if number_participants: participants_pages_prob_04.append(number_participants[i]*count_pages[i])
                if final_prediction[i] > 0.5:
                    positive_classified += 1
                    positive_classified_pages.append(count_pages[i])
                    if number_participants: positive_classified_participants_pages.append(number_participants[i]*count_pages[i])
                    if final_prediction[i] > 0.6:
                        documents_prob_06 += 1
                        documents_prob_pages_06.append(count_pages[i])
                        if number_participants: participants_pages_prob_06.append(number_participants[i]*count_pages[i])
                        if final_prediction[i] > 0.8:
                            documents_prob_08 += 1
                            documents_prob_pages_08.append(count_pages[i])
                            if number_participants: participants_pages_prob_08.append(number_participants[i]*count_pages[i])
        else:
            documents_prob_under_02 += 1
            documents_prob_under_pages_02.append(count_pages[i])
            if number_participants: participants_pages_prob_under_02.append(number_participants[i]*count_pages[i])

    sum_pages_positively_classified = sum(positive_classified_pages)
    sum_pages_over_08 = sum(documents_prob_pages_08)
    sum_pages_over_06 = sum(documents_prob_pages_06)
    sum_pages_over_04 = sum(documents_prob_pages_04)
    sum_pages_over_02 = sum(documents_prob_pages_02)
    sum_pages_under_02 = sum(documents_prob_under_pages_02)
    sum_participants_pages_positively_classified = sum(positive_classified_participants_pages)
    sum_participants_pages_over_08 = sum(participants_pages_prob_08)
    sum_participants_pages_over_06 = sum(participants_pages_prob_06)
    sum_participants_pages_over_04 = sum(participants_pages_prob_04)
    sum_participants_pages_over_02 = sum(participants_pages_prob_02)
    sum_participants_pages_under_02 = sum(participants_pages_prob_under_02)
    #from python 3.6 onwards a standard dict can be used
    from collections import OrderedDict
    report_dict = OrderedDict()
    report_dict['Total documents'] = len(batch_ids)
    report_dict['Positively Classified'] = positive_classified
    report_dict['Probability over 0.8'] = documents_prob_08
    report_dict['Probability over 0.6'] = documents_prob_06
    report_dict['Probability over 0.4'] = documents_prob_04
    report_dict['Probability over 0.2'] = documents_prob_02
    report_dict['Probability under 0.2'] = documents_prob_under_02
    report_dict['Pages Positively Classified'] = sum_pages_positively_classified
    report_dict['Pages Classified over 0.8'] = sum_pages_over_08
    report_dict['Pages Classified over 0.6'] = sum_pages_over_06
    report_dict['Pages Classified over 0.4'] = sum_pages_over_04
    report_dict['Pages Classified over 0.2'] = sum_pages_over_02
    report_dict['Pages Classified under 0.2'] = sum_pages_under_02
    if courses: report_dict['Number of courses'] = len(courses)
    if number_participants: 
        report_dict['Pages x Participants positively classified'] = sum_participants_pages_positively_classified
        report_dict['Pages x Participants over 0.8'] = sum_participants_pages_over_08
        report_dict['Pages x Participants over 0.6'] = sum_participants_pages_over_06
        report_dict['Pages x Participants over 0.4'] = sum_participants_pages_over_04
        report_dict['Pages x Participants over 0.2'] = sum_participants_pages_over_02
        report_dict['Pages x Participants under 0.2'] = sum_participants_pages_under_02

    report_name_json = 'report_custom_under_'+pages_limit+'.json'
    report_name_csv = 'report_custom_under_'+pages_limit+'.csv'
    with open(join(results_path, report_name_json), 'w') as fp:
        json.dump(report_dict, fp)
    with open(join(results_path,report_name_csv),'w') as fp:
        w = csv.writer(fp)
        w.writerows(report_dict.items())

if __name__ == "__main__":
    args = sys.argv
    print("Arguments: (1)Path to files (2)Path to result files  (3)Path to metadata file (4)Limit of pages")
    files_path = args[1]
    pdf_files = glob(join(files_path,"*.{}".format('pdf')))
    pool = Pool(1)
    res = pool.map(process, pdf_files)
    pool.close()
    pool.join() 
    res_fix={}
    for i,x in enumerate(res):
        i_d = splitext(basename(pdf_files[i]))[0]
        res_fix[i_d] = x 
    results_path = args[2]
    metadata_file = args[3]
    reader = csv.DictReader(open(metadata_file, 'r'))
    metadata = {}
    for row in reader:
        i_d = row['document_id']
        metadata[i_d] = {}
        metadata[i_d]['number_participants'] = row['number_participants']
        metadata[i_d]['course_name'] = row['course_name']
    if len(args) == 5:
        pages_limit = int(args[4])
    elif len(args) == 4: pages_limit = 250
    else: print("incorrect arguments.");sys.exit(1)
    result_files = glob(join(results_path,"result*.{}".format('json')))
    ids = []
    probabilities = []
    count_pages = []
    number_participants = []
    courses = []
    for rf in result_files:
        with open(rf, "r") as jsonFile:
            json_data = json.load(jsonFile)
        for k,v in json_data.items():
            ts = k.split('-')
            i_d = ts[0]
            typ = ts[1]
            if res_fix[i_d] <= pages_limit and i_d not in ids and typ != 'prediction':
                ids.append(i_d)
                probabilities.append(json_data[k])
                count_pages.append(int(res_fix[i_d]))
                number_participants.append(int(metadata[i_d]['number_participants']))
                courses.append(metadata[i_d]['course_name'])
    
    write_report(results_path, probabilities, ids, count_pages, courses, number_participants, str(pages_limit))

        