# -*- coding: utf-8 -*-

# Feature_Extractor.py

import os, sys
from os.path import join, realpath, dirname, isdir, basename, isfile

import csv, re, json
import pandas as pd
import numpy as np
from multiprocessing import Pool
from doc_globals import*

from PIL import Image as PI
from wand.image import Image

from bow_classifier.bow_classifier import BowClassifier
import subprocess

import cProfile, pstats

class Feature_Extractor():
    def __init__(self, bow_features, numeric_features, meta_path=join(DATA_PATH,"classified_metadata.csv")):
        self.metadata = pd.read_csv(meta_path,
            header=0,
            delimiter=',',
            quoting=1,
            encoding='utf-8')

        self.bow_features = bow_features
        self.numeric_features = numeric_features

        self.bow_classifiers = []
        for bf in bow_features:
            self.bow_classifiers.append(BowClassifier(bf))

    def extract_features(self, data, outfile, p=-1, profiling=False):

        fieldnames = self.bow_features
        fieldnames.extend(self.numeric_features)
        fieldnames.append("class")
        fieldnames.append("document_id")

        # Extract Features parallel
        if p == -1:
            pool = Pool()
        else:
            pool = Pool(p)
        if(not(profiling)):
            res = pool.map(self.get_data_vector, data)
        else:
            pr = cProfile.Profile()
            pr.enable()
            res = []
            for d in data:
                res.append(self.get_data_vector(d))
            pr.disable()
            pr.create_stats()
            ps = pstats.Stats(pr).sort_stats('tottime')
            ps.print_stats(0.1)
        
        with open(outfile,"w") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=',')
            writer.writeheader()
            for row in res:
                if(not(row is None)):
                    writer.writerow(row)

    def generate_error_features(self, features):
        #generates the error features and replaces the nan values
        #
        #@result:   list of features with the error features included
        
        error_feature = [0.0] * len(features)
        
        for i, x in enumerate(features):
            for j, e in enumerate(x):
                if e == 'nan':
                    error_feature[i] = 1.0
                    x[j] = 1.0                
        
        features = [x + [error_feature[i]] for i, x in enumerate(features)]
        return features

    def get_data_vector(self, t_data):

        feature_data = []
        doc_class = t_data[0]
        doc_id = t_data[1]

        filepath = join(PDF_PATH,doc_id+'.pdf')

        if(not(isfile(filepath))):
            print(print_bcolors(["WARNING","BOLD"],
                "doc with id: %s was not found!!!" %(doc_id,)))
            return None

        values_dict, bow_strings = self.get_num_vals_and_bow_strings(doc_id)
        if(values_dict is None):
            print(print_bcolors(["WARNING","BOLD"],
                "doc with id: %s is password protected!!!" %(doc_id,)))
            return None

        for bc in self.bow_classifiers:
            #try:
            values_dict[bc.name] = bc.get_function(bow_strings[bc.name],"custom_words_val")
            #except:
                # print(print_bcolors(["WARNING","BOLD"],
                #     "The BowClassifier for %s failed for id %s!" %(bc.name,doc_id)))
                # values_dict[bc.name] = np.nan

        values_dict["class"] = doc_class
        values_dict["document_id"] = doc_id

        return values_dict

    def get_num_vals_and_bow_strings(self, doc_id):

        values_dict = {}
        bow_strings = {}

        filepath = join(PDF_PATH,doc_id+'.pdf')
        pdfinfo_path = join(DATA_PATH,"lib_bow/pdfmetainfo.json")
        text_path = join(TXT_PATH,doc_id+".json")
        entropy_path = join(DATA_PATH,"lib_bow/pdf_grayvalue_entropy.json")

        # get pdfinfo dict information
        pdfinfo_data = None
        pdfinfo_dict = None
        if(isfile(pdfinfo_path)):
            with open(pdfinfo_path,"r") as f:
                pdfinfo_data = json.load(f)
            if(doc_id in pdfinfo_data):
                pdfinfo_dict = pdfinfo_data[doc_id]
            else:
                pdfinfo_dict = pdfinfo_get_pdfmeta(filepath)
                pdfinfo_data[doc_id] = pdfinfo_dict
                with open(pdfinfo_path,"w") as f:
                    json.dump(pdfinfo_data, f, indent=4)
        else:
            pdfinfo_data = {}
            pdfinfo_dict = pdfinfo_get_pdfmeta(filepath)
            pdfinfo_data[doc_id] = pdfinfo_dict
            with open(pdfinfo_path,"w") as f:
                json.dump(pdfinfo_data, f, indent=4)


        if(pdfinfo_dict is None):
            return None, None

        # sort infomation  into dicts
        if("author" in self.bow_features):
            bow_strings["author"] = pdfinfo_dict["author"]
        if("producer" in self.bow_features):
            bow_strings["producer"] = pdfinfo_dict["producer"]
        if("creator" in self.bow_features):
            bow_strings["creator"] = pdfinfo_dict["creator"]

        if("pages" in self.numeric_features):
            values_dict["pages"] = pdfinfo_dict["pages"]
        if("file_size" in self.numeric_features):
            values_dict["file_size"] = pdfinfo_dict["file_size"]
        if("page_rot" in self.numeric_features):
            values_dict["page_rot"] = pdfinfo_dict["page_rot"]
        if("page_size_x" in self.numeric_features):
            values_dict["page_size_x"] = pdfinfo_dict["page_size_x"]
        if("page_size_y" in self.numeric_features):
            values_dict["page_size_y"] = pdfinfo_dict["page_size_y"]

        # get csv metadata information
        try:
            csvmeta_series = self.metadata.loc[self.metadata['document_id'] == doc_id].reset_index(drop=True)   
            if("title" in self.bow_features):
                meta_string = csvmeta_series["title"].item()
                if(type(meta_string)!=str):
                    meta_string = None
                bow_strings["title"] = meta_string
            if("filename" in self.bow_features):
                meta_string = csvmeta_series["filename"].item()
                if(type(meta_string)!=str):
                    meta_string = None
                bow_strings["filename"] = meta_string
            if("folder_name" in self.bow_features):
                meta_string = csvmeta_series["folder_name"].item()
                if(type(meta_string)!=str):
                    meta_string = None
                bow_strings["folder_name"] = meta_string
            if("folder_description" in self.bow_features):
                meta_string = csvmeta_series["folder_description"].item()
                if(type(meta_string)!=str):
                    meta_string = None
                bow_strings["folder_description"] = meta_string
            if("description" in self.bow_features):
                meta_string = csvmeta_series["description"].item()
                if(type(meta_string)!=str):
                    meta_string = None
                bow_strings["description"] = meta_string

        except:
            print(print_bcolors(["WARNING","BOLD"], "No csv metadata for doc id %s!!!" %(doc_id,)))
            if("title" in self.bow_features):
                bow_strings["title"] = None
            if("filename" in self.bow_features):
                bow_strings["filename"] = None
            if("folder_name" in self.bow_features):
                bow_strings["folder_name"] = None
            if("folder_description" in self.bow_features):
                bow_strings["folder_description"] = None
            if("description" in self.bow_features):
                bow_strings["description"] = None
            

        # get content information
        images = 0
        text = ""
        if(not(isfile(text_path))):
            text, images = ghostcript_get_pdftext(filepath, first_page=-1, last_page=-1, save_json=True)
        else:
            with open(text_path,"r") as f:
                text_data = json.load(f)
            if(data is None):
                text = "password_protected"
                text_length = np.nan
            else:
                l_page = max(list(map(int, text_data.keys())))
                # concatenate the text
                for i in range(1,l_page+1):
                    page_text = text_data[str(i)]
                    if(len(page_text)<20):
                        images+=1
                    text += page_text
                text_length = float(len(text))/l_page
                images = float(images)/l_page

        if("text" in self.bow_features):
            bow_strings["text"] = text
        if("images" in self.numeric_features):
            values_dict["images"] = images
        if("copyright_symbol" in self.numeric_features):
            symbols = re.findall(r'DOI|ISBN|©', text)
            values_dict["copyright_symbol"] = len(symbols)>0
        if("word_count" in self.numeric_features):
            values_dict["word_count"] = text_length

        #TODO create entropy csvfile
        if("entropy" in self.numeric_features):
            entropy_dict = None
            if(isfile(entropy_path)):
                with open(entropy_path,"r") as f:
                    entropy_dict = json.load(f)
                if(doc_id in entropy_dict):
                    # (entropy,color) = entropy_dict[doc_id]
                    entropy = entropy_dict[doc_id]
                else:
                    entropy, color = get_grayscale_entropy_tmpfile(filepath,1,5)
                    entropy_dict[doc_id] = (entropy,color)
                    with open(entropy_path,"w") as f:
                        json.dump(entropy_dict, f, indent=4)
            else:
                entropy_dict = {}
                entropy, color = get_grayscale_entropy_tmpfile(filepath,1,5)
                entropy_dict[doc_id] = (entropy,color)
                with open(entropy_path,"w") as f:
                    json.dump(entropy_dict, f, indent=4)

            values_dict["entropy"]=entropy
            # values_dict["color"]=color


        return values_dict, bow_strings

    def train_bow_classifiers(self,filenames,classes):
        #function to train modules if needed. Each module called should have a train function
        #For now modules are pre-trained
        #We want a separate function for this
        for feat in self.bow_classifiers:
            feat.train(filenames,classes)


def pdfinfo_get_pdfmeta(fp):
    output = subprocess.Popen(["pdfinfo", fp],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE).communicate()[0].decode(errors='ignore')

    if(output==""):
        return None
    meta_dict = {}
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
            meta_dict[key] = None 
            continue
        if(key == "page_size"):
            val = val.split()
            meta_dict["page_size_x"] = float(val[0])
            key = "page_size_y"
            val = float(val[2])
        elif(key == "pages"):
            val = int(val)
        elif(key == "file_size"):
            val = val.split()[0]
            val = float(val)/1000
        elif(key == "page_rot"):
            val = int(val)>0
        elif(key == "encrypted"):
            val = not(val=="no")
        meta_dict[key] = val
    
    if not 'author' in meta_dict:
        meta_dict['author']= None
    if not 'creator' in meta_dict:
        meta_dict['creator']= None
    if not 'producer' in meta_dict:
        meta_dict['producer']= None
    if not 'title' in meta_dict:
        meta_dict['title']= None
        
    return meta_dict

def ghostcript_get_pdftext(fp, first_page=-1, last_page=-1, save_json=True):
    out = join(TXT_PATH,basename(fp)[:-4]+".json")

    args = ["gs", "-dNOPAUSE", "-dBATCH", "-sDEVICE=txtwrite", "-sOutputFile=-",fp]
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

    if(re.search(invalid_file_regex, output)):
        with open(out, 'w') as fp:
            json.dump(None, fp, indent=4)
        return None, 0

    output = re.sub(sub_regex, '', output)
    output = re.sub(loading_regex, '', output)
    output = re.sub(query_regex, '', output)
    output = re.sub(cant_find_regex, '', output)
    output = re.sub(didnt_find_regex, '', output)

    page_break_regex = "Page [0-9]+\n"
    pages = re.split(page_break_regex, output)

    txt = ""
    page_dict = {}
    empty_pages = 0
    for i in range(1, len(pages)):
        if(pages[i]==""):
            empty_pages +=1
        else:
            pages[i] = pages[i].split('done.\n')[-1]
        page_dict[i] = pages[i]
        txt += pages[i]

    with open(out, 'w') as fp:
        json.dump(page_dict, fp, indent=4)

    empty_pages = float(empty_pages)/(len(pages))
    return txt, empty_pages

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

    invalid_file_regex = r'Error: /invalidfileaccess in pdf_process_Encrypt'

    if(re.search(invalid_file_regex, output.decode())):
        return np.nan

    entropy = []
    color = False
    for i in range(first_page,last_page+1):
        imgfile = "tmp-%03d.jpeg"%(i,)
        if(not(isfile(imgfile))):
            continue
        pil_image = PI.open(imgfile)
        # with PI.open(io.BytesIO(page)) as pil_image:
        gs_image = pil_image.convert("L")
        hist = np.array(gs_image.histogram())
        hist = np.divide(hist,np.sum(hist))
        hist[hist==0] = 1
        e = -np.sum(np.multiply(hist, np.log2(hist)))
        entropy.append(e)

        if(not(color)):
            col_image = pil_image.convert('RGB')
            np_image = np.array(col_image)
            if((np_image[:,:,0]==np_image[:,:,1])==(np_image[:,:,1]==np_image[:,:,2])):
                color = True

        os.remove(imgfile)

    if(len(entropy)==0):
        print(print_bcolors(["WARNING","BOLD"],
            "Zero images loaded. Either pdf is empty or Ghostscript didn't create images correctly."))
        mean_entropy = np.nan
    else:
        mean_entropy = np.mean(entropy)
    return mean_entropy, color

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


if __name__ == "__main__":
    args = sys.argv
    len_args = len(args)

    usage = "python Feature_Extractor.py input=<datafile.csv> output=<output_file.csv> [num_cores=<number of cores>]\n"+ "- the input csv needs to contain document ids in the first and classifications in the second column\n"+\
    "- the output will contain columns for the features the classifications and the document ids in this order\n"+\
    "- the default number of cores is the maximum amount available"
    data = []

    if(not((len_args==3) or (len_args==4))):
        print(usage)
        sys.exit(1)

    data_file = args[1].split("=",1)[1]
    print(data_file)
    with open(data_file, 'r') as df:
        reader = csv.reader(df)
        data = list(reader)

    outfile = args[2].split("=",1)[1]

    if(len_args==4):
        try:
            p = int(args[3].split("=",1)[1])
        except ValueError:
            print("<number of cores> needs to be an integer")
            print(usage)
            sys.exit(1)
    else:
        p = -1

    bow_features = ["text","author","producer","creator","title","filename","folder_name","folder_description", "description"]
    bow_features = ["text","producer","creator","filename","folder_name"]
    numeric_features = ["pages", "file_size", "page_rot", "page_size_x", "page_size_y", "word_count", "copyright_symbol", "images", "entropy"]

    fe = Feature_Extractor(bow_features, numeric_features, join(DATA_PATH,"classified_metadata.csv"))

    fe.extract_features(data=data,outfile=outfile, p=p, profiling=True)