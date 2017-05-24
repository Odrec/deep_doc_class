import sys, os

from PIL import Image as PI
from wand.image import Image

import pyocr
import pyocr.builders
import io

from os.path import join, realpath, dirname, isdir, isfile
from PyPDF2 import PdfFileWriter, PdfFileReader

import numpy as np
from time import time

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage# if the argument is set to -1, all pages are read
try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO

import subprocess
import re
from doc_globals import* 

import json
import csv
import pandas as pd

import traceback

page_break_regex = "Page [0-9]+\n"

def pdfinfo_get_pdfmeta(fp):
    output = subprocess.Popen(["pdfinfo", fp], stdout=subprocess.PIPE).communicate()[0].decode()
    return output

def exiftool_get_pdfmeta(fp):
    output = subprocess.Popen(["exiftool", "-a", "-G1", "--System", fp], stdout=subprocess.PIPE).communicate()[0].decode()
    return output

def pypdf_get_pdfmeta(fp):
    pdf_toread = PdfFileReader(fp)
    
    if not pdf_toread.isEncrypted:
        pdf_info = pdf_toread.getDocumentInfo()
    else:
        #can't read metadata because file is encrypted
        print("encrypted")
        return np.nan

    return pdf_info

def pdfminer_get_pdftext(fp, num_pages=-1):
    try:
        rsrcmgr = PDFResourceManager()
        retstr = StringIO()
        codec = 'utf-8'
        laparams = LAParams()
        device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        password = ""
        maxpages = num_pages
        caching = True
        pagenos=set()
        for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):
            interpreter.process_page(page)

        text = retstr.getvalue()
        #fp.close()
        device.close()
        retstr.close()
    except:
        print('Text not extractable')
        text = ''
    return text

def ghostcript_get_pdftext(fp, first_page=-1, last_page=-1):
    args = ["gs", "-dNOPAUSE", "-dBATCH", "-sDEVICE=txtwrite", "-sOutputFile=-",fp]
    if(not(first_page==-1)):
        args.append("-dFirstPage=%d"%(first_page,))
    if(not(last_page==-1)):
        args.append("-dLastPage=%d"%(last_page,))
    output = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0].decode(errors='ignore')
    
    page_break_regex = "Page [0-9]+\n"
    pages = re.split(page_break_regex, output)
    if(len(pages)==1):
        print("Nothing found in %s!" % (fp,))
        return None, False
    try:
        gs_page_num = int(pages[0].split(" ")[-1][0:-2])
    except ValueError:
        print(pages[0])
        return None, False
    found_page_num = len(pages)-1
    # if(not(gs_page_num==found_page_num)):
    #     print("File: %s caused some error with the pages! gs found %d pages but %d where splitted." %  (fp, gs_page_num, found_page_num))
    #     return None
    # else:
    page_dict = {}
    empty_pages = 0
    for i in range(1, len(pages)):
        if(pages[i]==""):
            empty_pages +=1
        else:
            pages[i] = pages[i].split('done.\n')[-1]
        page_dict[i] = pages[i]

    if(empty_pages>0):
        print("Empty pages: %d" % (empty_pages,))
    else:
        print("Succes!")
    return page_dict, empty_pages>0

def get_grayscale_entropy(fp, first_page=-1, last_page=-1):
    args = ["gs", "-dNOPAUSE", "-dBATCH", "-sDEVICE=jpeg", "-sOutputFile=-"]
    if(not(first_page==-1)):
        args.append("-dFirstPage=%d"%(first_page,))
    if(not(last_page==-1)):
        args.append("-dLastPage=%d"%(last_page,))
    args.append("-r200")
    args.append(fp)

    output = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0]
    
    page_break_regex = b'Page [0-9]+\n'
    sub_regex = b'Substituting (.)+\n'
    loading_regex = b'Loading (.)+\n'
    query_regex = b'Querying (.)+\n'
    cant_find_regex = b'Can\'t find \(or can\'t open\) (.)+\n'
    didnt_find_regex = b'Didn\'t find (.)+\n'

    # output = re.sub(sub_regex, b'', output)
    # output = re.sub(loading_regex, b'', output)
    # output = re.sub(query_regex, b'', output)
    # output = re.sub(cant_find_regex, b'', output)
    # output = re.sub(didnt_find_regex, b'', output)
    # print(output)

    pages = re.split(page_break_regex, output)

    entropy = []
    for i in range(1, len(pages)):
        page = pages[i]
        page = page.split(b'done.\n')[-1]
        page = re.sub(sub_regex, b'', page)
        page = re.sub(loading_regex, b'', page)
        page = re.sub(query_regex, b'', page)

        pil_image = PI.open(io.BytesIO(page))
        # with PI.open(io.BytesIO(page)) as pil_image:
        gs_image = pil_image.convert("L")
        hist = np.array(gs_image.histogram())
        hist = np.divide(hist,np.sum(hist))
        hist[hist==0] = 1
        e = -np.sum(np.multiply(hist, np.log2(hist)))
        entropy.append(e)

    return np.mean(entropy)

def get_grayscale_entropy_tmpfile(fp, first_page=-1, last_page=-1):
    args = ["gs", "-dNOPAUSE", "-dBATCH", "-sDEVICE=jpeg", "-sOutputFile=tmp-%03d.jpeg"]
    if(not(first_page==-1)):
        args.append("-dFirstPage=%d"%(first_page,))
    if(not(last_page==-1)):
        args.append("-dLastPage=%d"%(last_page,))
    args.append("-r200")
    args.append(fp)

    output = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0]

    number_regex = b'[0-9]+'
    page_regex = b'pages [0-9]+ through [0-9]+'
    match = re.search(page_regex, output)
    match_string = match.group()
    numbers = re.findall(number_regex, match_string)

    fpage = int(numbers[0])
    lpage = int(numbers[1])+1

    entropy = []
    for i in range(fpage,lpage):
        imgfile = "tmp-%03d.jpeg"%(i,)
        if(not(isfile(imgfile))):
            print("Ghostscript didn't create image correctly")
            continue
        pil_image = PI.open(imgfile)
        # with PI.open(io.BytesIO(page)) as pil_image:
        gs_image = pil_image.convert("L")
        hist = np.array(gs_image.histogram())
        hist = np.divide(hist,np.sum(hist))
        hist[hist==0] = 1
        e = -np.sum(np.multiply(hist, np.log2(hist)))
        entropy.append(e)
        os.remove(imgfile)

    if(len(entropy)==0):
        print("Zero images loaded. Either pdf is emptyor Ghostscript didn't create images correctly.")
        mean_entropy = np.nan
    else:
        mean_entropy = np.mean(entropy)
    return mean_entropy

def get_pil_image_from_pdf(fp, page=1):
    args = ["gs", "-dNOPAUSE", "-dBATCH", "-sDEVICE=jpeg", "-sOutputFile=-", "-r200"]
    args.append("-dFirstPage=%d"%(page,))
    args.append("-dLastPage=%d"%(page,))
    args.append(fp)

    output = subprocess.Popen(args, stdout=subprocess.PIPE).communicate()[0]
    page_break_regex = b'Page [0-9]+\n'
    page = re.split(page_break_regex, output)[-1]
    page = page.split(b'done.\n')[-1]
    pil_image = PI.open(io.BytesIO(page))
    return pil_image

def get_text_from_pil_img(pil_image, lang="deu"):
    if(not(lang in ["eng", "deu", "fra"])):
        print("Not the right language!!!\n Languages are: deu, eng, fra")

    tool = pyocr.get_available_tools()[0]
    txt = tool.image_to_string(
        pil_image,
        lang=lang,
    )
    return txt

def time_comparison(filepath, first_page, last_page):
    filepointer = open(filepath,'rb')

    start = time()
    m1 = pdfinfo_get_pdfmeta(filepath)
    pdfinfo = time()-start
    print("pdfinfo %0.3fs" % (pdfinfo, ))

    start = time()
    m2 = exiftool_get_pdfmeta(filepath)
    exiftool = time()-start
    print("exiftool %0.3fs" % (exiftool, ))

    start = time()
    m3 = pypdf_get_pdfmeta(filepointer)
    pypdf = time()-start
    print("pypdf %0.3fs" % (pypdf, ))

    start = time()
    t1 = ghostcript_get_pdftext(filepath, first_page=first_page, last_page=last_page)
    ghostscript = time()-start
    print("ghostscript %0.3fs" % (ghostscript, ))

    # start = time()
    # t2 = pdfminer_get_pdftext(filepointer, num_pages=last_page-first_page+1)
    # pdfminer = time()-start
    # print("pdfminer %0.3fs" % (pdfminer, ))

    start = time()
    e1 = get_grayscale_entropy(filepath, first_page=first_page, last_page=last_page)
    entropy_pipe = time()-start
    print("entropy_pipe %0.3fs" % (entropy_pipe, ))

    start = time()
    get_grayscale_entropy_tmpfile(join(PDF_PATH,"5bb2af5fa13c2c99e4cb1741b118cecf.pdf"),first_page=first_page, last_page=last_page)
    entropy_tmpfile = time()-start
    print("entropy_tmpfile %0.3fs" % (entropy_tmpfile, ))
        

    start = time()
    pil_image = get_pil_image_from_pdf(filepath, page=first_page)
    get_image = time()-start
    print("get_image %0.3fs" % (get_image, ))

    start = time()
    t3 = get_text_from_pil_img(pil_image, lang="deu")
    ocr_text = time()-start
    print("ocr_text %0.3fs" % (ocr_text, ))

def pdf_to_json_pypdf(fp):
    json_dict = {}
    inputpdf = PdfFileReader(fp,strict=False)
    num_pages = inputpdf.getNumPages()
    for i in range(0,num_pages):
        json_dict[i] = inputpdf.getPage(i).extractText()

def extract_text(filenames, target,source,pages = 1):
    """

    :param filenames: list of pdffiles to extract pdfs from
    :param target:    targetfolder
    :param source:    source folder
    :param pages:     number of pages per file
    :return:          True if done, else False
    """
    b = TextScore()
    global c
    for f in filenames:
        try:
            c += 1
            print(str(c)+'/'+str(len(filenames)),end='\r')
            fp = open(source+f+'.pdf','rb')
            tf = open(target+f+'.txt','w+')
            txt = b.convert_pdf_to_txt(fp,pages)
            tf.write(txt)
        except:
            e = sys.exc_info()[0]
            print(e)
            fp.close()
            tf.close()
            #print("Troubleshooting")
    return True

def convert_all_pdf_to_text(source=join(DATA_PATH, "pdf_files"), target=join(DATA_PATH, "json_txt_files")):
    if(isdir(source)):
        filenames = os.listdir(source)
    else:
        print("source not a directory!")
        sys.exit(1)

    if(not(isdir(target))):
        os.mkdir(target)

    counter = 0
    empty_cnt = 0
    next_perc = 0.01
    for filename in filenames:
        filepath = join(source,filename)
        page_dict, empty = ghostcript_get_pdftext(filepath, first_page=-1, last_page=-1)
        if(empty):
            empty_cnt +=1
        json_path = join(target, filename[:-3]+"json")
        counter += 1
        if(counter/len(filenames)>next_perc):
            print("%.2f %% done" % (next_perc*100))
            next_perc += 0.01

        with open(json_path, 'w') as fp:
            json.dump(page_dict, fp, indent=4)

    print("Done!!!")
    print("%.2f %% had empty pages!" % (empty_cnt/len(filenames)*100))

def get_new_features(doc_ids,classes,target_file="new_features.csv"):
    json_path = join(DATA_PATH, "json_txt_files")
    fieldnames = ["copyright_sign", "words_per_page", "perc_txt_pages", "entropy"]


    counter = 0
    next_perc = 0.01

    copyright_signs = [u'\u00a9',"ISBN","DOI"]
    with open(join(FEATURE_VALS_PATH, target_file),"w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=',')
        writer.writeheader()
        for idx in range(len(doc_ids)):
            filename = doc_ids[idx]
            classification = classes[idx]
            filepath = join(json_path,filename+".json")
            rowdict = {}
            try:
                data = json.load(open(filepath,"r"))
                if(data==None):
                    rowdict["copyright_sign"] = 0.0
                    rowdict["words_per_page"] = np.nan
                    rowdict["perc_img_pages"] = np.nan
                    rowdict["entropy"] = np.nan
                    rowdict["class"] = classification
                    rowdict["document_id"] = filename
                    writer.writerow(rowdict)
                else:
                    word_counts = []
                    found_cp_sign = False
                    for page_key in data.keys():
                        txt = data[page_key]
                        if(not(txt=="")):
                            words = txt.split()
                            word_counts.append(len(words))
                            for word in words:
                                if word in copyright_signs:
                                    found_cp_sign = True
                                    break;
                        else:
                            word_counts.append(0.0)

                            # if(u'\u00a9' in words or "ISBN" in words or "DOI" in words):
                            #     found_cp_sign = True
                    words_per_page = np.mean(word_counts)
                    perc_img_pages = (len(data.keys())-len(word_counts))/len(data.keys())
                    entropy = get_grayscale_entropy_tmpfile(join(PDF_PATH,filename+".pdf"), first_page=1, last_page=min(len(data.keys()), 5))
                    
                    rowdict["copyright_sign"] = float(found_cp_sign)
                    rowdict["words_per_page"] = words_per_page
                    rowdict["perc_img_pages"] = perc_img_pages
                    rowdict["entropy"] = entropy
                    rowdict["class"] = classification
                    rowdict["document_id"] = filename
                    writer.writerow(rowdict)

            except FileNotFoundError:
                print("file %s not found adding nans!" %(filename,))
                rowdict["copyright_sign"] = 0.0
                rowdict["words_per_page"] = np.nan
                rowdict["perc_img_pages"] = np.nan
                rowdict["entropy"] = np.nan
                rowdict["class"] = classification
                rowdict["document_id"] = filename
                writer.writerow(rowdict)

            except:
                traceback.print_exc()
                print(filename)
                sys.exit(1)

            counter += 1
            if(counter/len(filenames)>next_perc):
                print("%.2f %% done" % (next_perc*100))
                next_perc += 0.01

def load_filenames_from_feature_file(features_file):
    features=pd.read_csv(features_file, header=0, delimiter=',', quoting=1, encoding='utf-8')
    filenames = features["document_id"].tolist()
    return filenames

def merge_feature_files(f1,f2,out):
    features1=pd.read_csv(f1, header=0, delimiter=',', quoting=1, encoding='utf-8')
    
    f_names = list(features1)
    np_features1 = features1.as_matrix(f_names[:-2])
    np_classes = np.array(features1[f_names[-2]].tolist())
    filenames = features1[f_names[-1]].tolist()

    features2=pd.read_csv(f2, header=0, delimiter=',', quoting=1, encoding='utf-8')
    
    f_names2 = list(features2)
    np_features2 = features2.as_matrix(f_names2)

    print(np.shape(np_features1))
    print(np.shape(np_features2))
    print(np.shape(np_classes))
    print(np.shape(filenames))

    combined_data = np.append(np_features1,np_features2,axis=1)
    print(np.shape(combined_data))
    combined_data = np.append(combined_data,np.array(np_classes).reshape(len(np_classes),1),axis=1)
    combined_data = np.append(combined_data,np.array(filenames).reshape(len(filenames),1),axis=1)

    combined_names = f_names[:-2]
    combined_names.extend(f_names2)
    combined_names.append(f_names[-2])
    combined_names.append(f_names[-1])

    print(np.shape(combined_data))
    print(len(combined_names))

    with open(join(FEATURE_VALS_PATH, out), 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=combined_names, delimiter=",")
        writer.writeheader()
        for row in combined_data:
            rowdict = {}
            for i, fname in enumerate(combined_names):
                rowdict[fname] = row[i]
            writer.writerow(rowdict)

def merge_feature_files(f1,f2,pos1,pos2,out):
    features1=pd.read_csv(f1, header=0, delimiter=',', quoting=1, encoding='utf-8')
    f_names = list(features1)
    np_features1 = features1.as_matrix(f_names[:-2])
    np_classes = np.array(features1[f_names[-2]].tolist())
    filenames = features1[f_names[-1]].tolist()

    features2=pd.read_csv(f2, header=0, delimiter=',', quoting=1, encoding='utf-8')
    f_names2 = list(features2)
    np_features2 = features2.as_matrix(f_names2[:-2])

    for p1,p2 in zip(pos1,pos2):
        np_features1[:,p1] = np_features2[:,p2]
    np_features1 = np.append(np_features1,np.array(np_classes).reshape(len(np_classes),1),axis=1)
    np_features1 = np.append(np_features1,np.array(filenames).reshape(len(filenames),1),axis=1)

    with open(join(FEATURE_VALS_PATH, out), 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=f_names, delimiter=",")
        writer.writeheader()
        for row in np_features1:
            rowdict = {}
            for i, fname in enumerate(f_names):
                rowdict[fname] = row[i]
            writer.writerow(rowdict)

if __name__ == "__main__":
    doc_path = '/home/kai/Workspace/doc_class/deep_doc_class/data/pdf_files'

    pres_pdf = "fc8278775cf12fdf84e88493f8efc645.pdf"
    pw_pdf = "8aa4e4c4ae4f63d9a382c1a9c74eb42f.pdf"
    norm_pdf = "00d0577a546dee16d43d04dbe490711e.pdf"
    norm_pdf2 = "0b6b1cb9c9cd832bf02cf67bf8302e2c.pdf"
    gray_test_pdf = "5bb2af5fa13c2c99e4cb1741b118cecf.pdf"
    pw_protected = "0f7222aeded1cc52d151fdb58d48085a.pdf"
    copyright = "0b6b1cb9c9cd832bf02cf67bf8302e2c.pdf"

    error = "d98d9c44a48ef8615328b5693f859236.pdf"

    # json_path = join(DATA_PATH, "json_txt_files")

    # filenames = load_filenames_from_feature_file(join(FEATURE_VALS_PATH, "whole_features_17_11.csv"))
    # get_new_features(filenames)

    # get_new_features([norm_pdf[:-4],gray_test_pdf[:-4],pw_protected[:-4],copyright[:-4], pres_pdf[:-4], "0d0577a546dee16d43d04dbe490711e"])
    

    # pil_image = get_pil_image_from_pdf(join(doc_path, image_test_pdf), page=1)
    # txt = get_text_from_pil_img(pil_image, lang="deu")
    # print(txt)

    # time_comparison(join(doc_path, gray_test_pdf), 1, 5)


    merge_feature_files(join(FEATURE_VALS_PATH,"whole_features_30_11.csv"),join(FEATURE_VALS_PATH,"metacsv_features_1_12.csv"),[9,11,12,13,14],[0,1,2,3,4],"whole_features_1_12.csv")


