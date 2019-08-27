#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 11:44:59 2017

@author: odrec
"""

from subprocess import call
import sys, os, csv
from glob import glob
from PIL import Image
import math
from os.path import basename, join, splitext, isfile, isdir, split
from os import makedirs

def convert_pdf_jpg(file, num_pages=3, slices=False):
    file_tuple = split(file)
    f_id = splitext(file_tuple[1])[0]
    files_path = file_tuple[0]
    images_path = join(files_path,'images/')
    if not isdir(images_path): makedirs(images_path)
    images_path = join(images_path, f_id)
    if not isdir(images_path): makedirs(images_path)
    output_jpg = join(images_path,f_id+'-1.jpg')
    if not isfile(output_jpg):
        call(["gs", "-dPDFFitPage", "-dNOPAUSE", "-sDEVICE=jpeg", "-dDEVICEWIDTH=250", 
        "-dDEVICEHEIGHT=250", "-dFirstPage=1", "-dLastPage=" + str(num_pages), 
        "-sOutputFile=" + images_path + "/" + f_id + "-%d.jpg", "-dJPEGQ=100", "-r300", 
        "-q", file, "-c", "quit"])
    images_paths = glob(join(images_path,"*.{}".format('jpg')))
    if slices: generate_image_slices(images_path)
    return images_paths, f_id
    
def generate_image_slices(images_path):
    files = []
    for root, dirs, fls in os.walk(images_path):
        for file in fls:
            files.append(file)
         
    for f in files:
        file_name = splitext(f)[0]
        img = Image.open(join(images_path,f))
        width, height = img.size

        horizontal_slice = int(math.ceil(height*0.2))
        
        upper_box = (0, 0, width, horizontal_slice)
        upper_slice = img.crop(upper_box)
        upper_slice.save(join(images_path, "1_" + file_name + ".jpg"))
        
        lower_box = (0, height-horizontal_slice, width, height)
        lower_slice = img.crop(lower_box)
        lower_slice.save(join(images_path, "2_" + file_name + ".jpg"))
        
        vertical_slice = int(math.ceil(width*0.2))
        
        left_box = (0, 0, vertical_slice, height)
        left_slice = img.crop(left_box)
        left_slice.save(join(images_path, "3_" + file_name + ".jpg"))
        
        right_box = (width-vertical_slice, 0, width, height)
        right_slice = img.crop(right_box)
        right_slice.save(join(images_path, "4_" + file_name + ".jpg"))
        
        image_center = int(math.ceil(width/2))
        center_slice = int(math.ceil(width*0.1))
        
        center_box = (image_center-center_slice, 0, image_center+center_slice, height)
        center_slice = img.crop(center_box)
        center_slice.save(join(images_path, "5_" + file_name + ".jpg"))
        
def check_images(file, overwrite=False, num_pages=3):
    file_data = split(file)
    file_path = file_data[0]
    file_name = file_data[1]
    images_path = join(file_path,'images/')
    if not isdir(images_path): return False
    file_images_path = join(images_path,file_name)
    if not isdir(file_images_path): return False
    images_paths = glob(join(file_images_path+"*.{}".format('jpg')))
    if overwrite:
        for i in images_paths: os.remove(i)
        images_paths = []
    if not images_paths or len(images_paths) < num_pages: return False
    return images_paths
    
if __name__ == "__main__":
    args = sys.argv
    la = len(args)
    usage = "Usage: pdftojpg.py [csv file with list of files to convert] \
    [-fp path to pdf file(s)] [-p number of pages to convert]"
    
    if (not la == 2 and not la == 4 and not la == 6) or '-h' in args or '--help' in args:
        print("Incorrect arguments.",la)
        print(usage)
    else:
        if '-p' in args:
            index_p = args.index('-p')+1
            if args[index_p].isdigit():
                num_pages = args[index_p]
                print("Using the first %s pages of each file."%num_pages)
            else: print(usage)
        else:
            num_pages = 1000000
            print("Using all the pages for each file.")
        
        f = args[1]
        file_list = []
        if isfile(f) and splitext(basename(f))[1] == '.csv':
            files = f
            if '-fp' in args:
                index_fp = args.index('-fp')+1
                if isdir(args[index_fp]):
                    with open(f) as cf:
                        data = csv.DictReader(cf, delimiter=';')
                        data_list = data
                        for d in data:
                            line = d['document_id']+'.pdf'
                            for file in os.walk(args[index_fp]):
                                for name in file:
                                    if line in name:
                                        file_list.append(join(args[index_fp],line))
                else:
                    print("You need to specify a path to the files with the -fp parameter.")
                    print(usage)
            else:
                print("The csv file with the list of files is not valid.")
                print(usage)                                    
        else:
            if '-fp' in args:
                index_fp = args.index('-fp')+1
                if isfile(args[index_fp]):
                    if splitext(basename(args[index_fp]))[1] == '.pdf':
                        file_list.append(args[index_fp])
                    else: print(usage)
                elif isdir(args[index_fp]):
                    files_list = glob(join(args[index_fp],"*.{}".format('pdf')))
                else: print(usage)
        
        for f in file_list:
            convert_pdf_jpg(f, num_pages)
