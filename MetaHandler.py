"""
MetaHandler.py

Offers different versions of loading the metadata.
Each functions has an own documentation.
Has some functions for evaluation of trial runs as well.

The main runs an example.

@author: kai
"""
# imports
import os, sys, io
from os.path import join, realpath, dirname, isdir
from datetime import datetime
import random
import numpy as np

import csv

# https://www.decalage.info/python/html for installation
import HTML

def get_whole_metadata(metafile):
    """
    Loads only the classified metadata

    @param metafile: the file containing the metadata or the full path as string
    @dtype metafile: string or file

    @return metadata: the whole metadata
    @dtype metadata: dict(document_id:list(metadata))
    """
    # Make sure the metafiel parameter is of the right type
    if(isinstance(metafile,str)):
        try:
            meta_csv = open(metafile)
        except IOError:
            print("The metafile attribute has to be a the filepath or a file opject!")
            sys.exit(0)
    else:
        try:
            assert(isinstance(metafile,io.IOBase))
            meta_csv = metafile
        except AssertionError:
            print("The metafile attribute has to be a the filepath or a file opject!")
            sys.exit(0)
    # Open the csv file
    reader = csv.DictReader(meta_csv)

    data_fname2idx = {}

    # get the fieldnames and parse them in the index dict
    data_fnames = reader.fieldnames
    for i in range(0,len(data_fnames)):
        data_fname2idx[data_fnames[i]] = i

    # make space for meta_data
    metadata = {}
    # read the data row by row
    for row in reader:
        # get the row ordered as the fieldnames
        datarow = [row[name] for name in data_fnames]
        # make ints to ints
        datarow[data_fname2idx["filesize"]] = int(datarow[data_fname2idx["filesize"]])
        # bools to bools
        datarow[data_fname2idx["author_dr"]] = bool(int(datarow[data_fname2idx["author_dr"]]))
        datarow[data_fname2idx["author_prof"]] = bool(int(datarow[data_fname2idx["author_prof"]]))
        datarow[data_fname2idx["is_pdf"]] = bool(int(datarow[data_fname2idx["is_pdf"]]))
        # and timestamps to timestampts
        datarow[data_fname2idx["upload_timestamp"]] = datetime.utcfromtimestamp(float(datarow[data_fname2idx["upload_timestamp"]]))
        # add the datarow
        metadata[datarow[data_fname2idx["document_id"]]] = datarow

    # close file if opened by this function
    if(isinstance(metafile,str)):
        meta_csv.close()
    # return the data
    return metadata

def get_classified_metadata(metafile, classfile, only_copyright_class=True):
    """
    Loads only the metadata of the documents which have been classified and their classification

    @param metafile: the file containing the metadata or the full path as string
    @dtype metafile: string or file

    @param classfile: the file containing the classifications or the full path as string
    @dtype classfile: string or file

    @param only_copyright_class: flag deciding if the classifications should only contain the field identifying
                                the copyright status or all the other information as well.
    @dtype only_copyright_class: boolean

    @return metadata: the metadata as dict
    @dtype metadata: dict(document_id:list(metadata))

    @return classifications: the classification as dict
    @dtype classifications: dict(document_id:classification)
    """

    ### Classifications ###
    # Make sure the classfile parameter is of the right type
    if(isinstance(classfile,str)):
        try:
            class_csv = open(classfile)
        except IOError:
            print("The metafile attribute has to be a the filepath or a file opject!")
            sys.exit(0)
    else:
        try:
            assert(isinstance(classfile,io.IOBase))
            class_csv = classfile
        except AssertionError:
            print("The metafile attribute has to be a the filepath or a file opject!")
            sys.exit(0)

    # open the csv reader
    reader = csv.DictReader(class_csv)

    # get all the fielnames and write them to the name->index dict
    class_fname2idx = {}
    class_fnames = reader.fieldnames
    # fieldnames are all in one string seperated by semicolons (why?)
    class_fnames = class_fnames[0].split(';')
    for i in range(0,len(class_fnames)):
        class_fname2idx[class_fnames[i]] = i

    # get classifications row by row
    classifications = {}
    for row in reader:
        # fields are one string seperated by semicolons (why?)
        datarow = list(row.values())[0].split(';')
        if(only_copyright_class):
            classifications[datarow[class_fname2idx["document_id"]]] = datarow[class_fname2idx["published"]]=="True"
        else:
            datarow[class_fname2idx["published"]] = datarow[class_fname2idx["published"]]=="True"
            classifications[datarow[class_fname2idx["document_id"]]] = datarow
    # close the file if open by this function
    if(isinstance(classfile,str)):
        class_csv.close()

    ### Metadata ###
    # Make sure the metafile parameter os of the right type
    if(isinstance(metafile,str)):
        try:
            meta_csv = open(metafile)
        except IOError:
            print("The metafile attribute has to be a the filepath or a file opject!")
            sys.exit(0)
    else:
        try:
            assert(isinstance(metafile,io.IOBase))
            meta_csv = metafile
        except AssertionError:
            print("The metafile attribute has to be a the filepath or a file opject!")
            sys.exit(0)

    # open the csv reader
    reader = csv.DictReader(meta_csv)

    # get the fieldnames and parse them in the name->index dict
    data_fname2idx = {}
    data_fnames = reader.fieldnames
    for i in range(0,len(data_fnames)):
        data_fname2idx[data_fnames[i]] = i

    # make space for meta_data
    metadata = {}
    # read the data row by row
    for row in reader:
        # if it has been classified
        if(row["document_id"] in classifications):
            # get the row ordered as the fieldnames
            datarow = [row[name] for name in data_fnames]
            # make ints to ints
            datarow[data_fname2idx["filesize"]] = int(datarow[data_fname2idx["filesize"]])
            # bools to bools
            datarow[data_fname2idx["author_dr"]] = bool(int(datarow[data_fname2idx["author_dr"]]))
            datarow[data_fname2idx["author_prof"]] = bool(int(datarow[data_fname2idx["author_prof"]]))
            datarow[data_fname2idx["is_pdf"]] = bool(int(datarow[data_fname2idx["is_pdf"]]))
            # and timestamps to timestampts
            datarow[data_fname2idx["upload_timestamp"]] = datetime.utcfromtimestamp(float(datarow[data_fname2idx["upload_timestamp"]]))
            # add the datarow
            metadata[row["document_id"]] = datarow

    # close the file if open by this function
    if(isinstance(meta_csv,str)):
        meta_csv.close()

    #It eliminates files that can be processed by all the modules
    #since the modules we have now don't use the metadata file
    #Creates a bug with the classes list during training since it
    #fails to index some files
    #    
    # throw out class ids that are not in the metadata
    #class_ids = classifications.keys()
    #for c_id in list(class_ids):
    #    if(not(c_id in metadata)):
    #        del classifications[c_id]

    # return metadata and classification
    return metadata, classifications

def gen_train_test_split(document_ids, test_size):
    '''
    Splits th provided data into training- and testsplits

    @param  document_ids: The document ids of the classified data
    @type   document_ids: list(string)

    @param  test_size: The fraction of the data supposed to be testing-data
    @type   test_size: float
    '''
    n_data = len(document_ids)
    n_test = int(n_data*test_size)
    doc_copy = document_ids[:]
    random.shuffle(doc_copy)
    train_doc = doc_copy[:n_test]
    test_doc = doc_copy[n_test:]

    return train_doc, test_doc

def result_html_confusiontable(filepath, targets, results):
    """
    Creates a confusiontable of the results and targets and writes it into an HTML file

    @param filepath: the whole path to the file that is to be created (should include .html at the end)
    @dtype filepath: string

    @param targets: the target values
    @dtype targets: list(bool)

    @param results: the result values
    @dtype results: list(bool)
    """
    try:
        f = open(filepath, 'w')
        f.write(html_intro("Doc_Result_Confusiontable"))
        f.write("<style>\n")
        f.write(".floating-box {\n\tdisplay: inline-block;\n\tmargin-top: 20px;\n\tmargin-bottom: 20px;\n\tborder: 3px solid #000000;}\n")
        f.write("table, td, th {\n\tborder: 3px solid black;}\n")
        f.write("table {\n\tborder-collapse: collapse;\n\twidth: 60%;}\n")
        f.write("td {\n\tbackground-color: #f2f2f2;\n\tborder: 2px solid black;\n\tvertical-align: center;\n\ttext-align: center;}\n")
        f.write("</style>\n")
        f.write("<center>\n")
        f.write(html_heading(3, "Document Copyright Classification\n<br>\nConfusiontable"))
        conf_data = get_conftable_data(targets, results)
        f.write(html_confusiontable(conf_data))
        f.write(html_end())
        f.close()
    except:
            print("Error during writing the HTML file!")
            sys.exit(0)

def html_confusiontable(data):
    """
    Create a HTML confusiontable with the data of an confusiontable

    @param vals: the data that have to be written into the table
    @dtype vals: list(3x3)

    @return t: the table
    @dtype t: string
    """
    # start table
    t = HTML.Table()
    # add header rows
    t.rows.append(HTML.TableRow([HTML.TableCell("Confusion\n<br>\nTable", header=True, attribs={"rowspan":"2","colspan":"2","width":"20%"})],
        attribs={"height":"10px"}))
    t.rows[-1].cells.append(HTML.TableCell("Acutal", header=True, attribs={"colspan":"2","width":"70%"}))
    t.rows[-1].cells.append(HTML.TableCell("Total", header=True, attribs={"rowspan":"2","width":"10%"}))

    t.rows.append(HTML.TableRow([HTML.TableCell("True", header=True)], attribs={"height":"10px"}))
    t.rows[-1].cells.append(HTML.TableCell("False", header=True))
    # add data rows
    t.rows.append(HTML.TableRow([HTML.TableCell("Predicted", attribs={"rowspan":"2"}, header=True),
        HTML.TableCell("True", header=True),
        HTML.TableCell(data[0,0]),HTML.TableCell(data[0,1]),
        HTML.TableCell(data[0,2])], attribs={"height":"50px"}))
    t.rows.append(HTML.TableRow([HTML.TableCell("False", header=True, attribs={"height":"50px"}),
        HTML.TableCell(data[1,0]),
        HTML.TableCell(data[1,1]),
        HTML.TableCell(data[1,2])]))
    t.rows.append(HTML.TableRow([HTML.TableCell("Total", attribs={"colspan":"2"}, header=True),
        HTML.TableCell(data[2,0]),
        HTML.TableCell(data[2,1]),
        HTML.TableCell(data[2,2])], attribs={"height":"10px"}))

    return str(t)

def html_intro(page_title):
    """
    Create the top of an html page with a page title

    @param page_title: the page title
    @dtype page_title: string

    @return intro: the html intro with the title
    @dtype intro: string
    """
    intro = "<!DOCTYPE html>\n<html>\n<head>\n<title>%s</title>\n</head>\n<body>\n" %(page_title)
    return intro

def html_end():
    """
    Create a html ending

    @return end: the html ending
    @dtype end: string
    """
    end = "</body>\n</html>\n"
    return end

def html_heading(num, heading):
    """
    Create different headings for a html page

    @param num: the thickness level of the heading
    @dtype num: int

    @param heading: the heading text
    @dtype heading: string

    @return heading: the html heading
    @dtype heading: string
    """
    heading = "<h%d>%s</h%d>\n"% (num,heading,num)
    return heading

def get_conftable_data(vals, targets, op=None,):
    """
    Test a set of values at a choosable operator.
    Provide statistics about how the truth values of this comparison translate to the classification

    @param vals: the data examples that are to be tested. For binary operator the second dimension has to be two
    @dtype vals: list

    @param op: the operator with which to compare the data. Its a unary or binary operator.
    @dtype op: operator

    @param targets: the target values
    @dtype targets: list

    @return ttable_counter: the table
    @dtype ttable_counter: np.array
    """

    ttable_counter = np.zeros((3,3))
    if(not(op is None)):
        op_args = len(inspect.getargspec(op)[0])
        if(op_args>2): raise ValueError('Too many arguments for the operator 2 maximum')

    # go through the values and count  up the stats
    for i in range(len(vals)):
        test_con = None
        if(op is None):
            test_con = vals[i]
        else:
            if(op_args==1):
                test_con = op(vals[i])
            else:
                test_con = op(vals[i,0],vals[i,1])

        if(test_con):
            ttable_counter[0][2]+=1
            if(targets[i]):
                ttable_counter[2][0]+=1
                ttable_counter[0][0]+=1
            else:
                ttable_counter[2][1]+=1
                ttable_counter[0][1]+=1
        else:
            ttable_counter[1][2]+=1
            if(targets[i]):
                ttable_counter[2][0]+=1
                ttable_counter[1][0]+=1
            else:
                ttable_counter[2][1]+=1
                ttable_counter[1][1]+=1
        ttable_counter[2][2]+=1
    return ttable_counter

if __name__ == '__main__':
    # the module path is the path to the project folder
    # beeing the parent folder of the folder of this file
    # MODUL_PATH = join(dirname(realpath(__file__)), os.pardir)
    MODUL_PATH = "/home/kai/Workspace/doc_class"
    # this is the path to the doc_data
    DOC_DATA_PATH = join(MODUL_PATH, "doc_data")
    metafile = join(DOC_DATA_PATH, "metadata.csv")
    classfile = join(DOC_DATA_PATH, "classification.csv")

    # metadata = get_whole_metadata(metafile)
    # print(len(metadata))
    metadata, classifications = get_classified_metadata(metafile, classfile, only_copyright_class=True)
    print(len(metadata))
    print(len(classifications))
    train_doc, test_doc = gen_train_test_split(classifications.keys(), 0.5)
    print(len(train_doc))
    print(len(test_doc))
    targets = classifications.values()

    DOC_ANALYSIS = join(MODUL_PATH, "doc_analysis")
    if(not(isdir(DOC_ANALYSIS))): os.makedirs(DOC_ANALYSIS)
    out_file = join(DOC_ANALYSIS, "test_conf_table.html")

    random_resuls = [random.random() for i in range(0,len(targets))]
    random_resuls = [rr>0.5 for rr in random_resuls]

    result_html_confusiontable(out_file,targets,random_resuls)

