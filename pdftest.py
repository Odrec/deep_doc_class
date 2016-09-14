# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 17:41:50 2016

simple sandbox script to test the capabilities of a pdf extractor i found

using pdfminer
using pypdf
using ghostscript
using ImageMagick Wand (including python wrapper)

the code and the used packages of this script show the possible solutions of
extracting data-vectors from pdfs.

note on image processing: it is actually possible to extract color info
(srgb) by manual array-like indexing from the image objekt
This makes it possible to extract the data Vector without using the hard
drive for saving the jpeg

@author: Mats
"""

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from cStringIO import StringIO
from wand.image import Image
from wand.color import Color
import os, os.path, sys

import tempfile
from wand.image import Image

# @param pages number of pages to transform to text starting from the first
# if the argument is set to -1, all pages are read 
# @return a utf-8 coded string from the pdf. 
def convert_pdf_to_txt(path,pages=-1):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    fp = file(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos=set()
    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):
        if(pages==0):
            break
        interpreter.process_page(page)
        pages -= 1

    text = retstr.getvalue()
    fp.close()
    device.close()
    retstr.close()
    return text

# @param source_file a string path to the source file
# @param target_file a string path to the target file
# desr_width width of the picture in pixels
# dest_height height of the picture in pixels
# pages: number of pages
# @return always true
#
# the function saves, starting from the first side, all pages of a given pdf
# as jpeg-files. the name of the saved file is given by he target file argument
# in case of multiple pages the function will auto-generate name-afixes so no
# data is lost in the process
#
# for application i would recommend to return the image object, which can be
# indexed like a dimensional array for extracting the srgb-values and building 
# the data-vector in the process.  
def pdf2jpg(source_file, target_file, dest_width=1895, dest_height=1080,pages=1):
    RESOLUTION    = 300
    ret = True
    img = Image(filename=source_file, resolution=(RESOLUTION,RESOLUTION))
    img.background_color = Color('white')
    img_width = img.width
    ratio     = float(dest_width) / float(img_width)
    img.resize(dest_width, int(ratio * img.height))
    img.format = 'jpeg'
    img.alpha_channel=False
    for im in img.sequence:
        if(pages==0):
            break
        pages -= 1
        ima = Image(image=im)
        ima.save(filename = target_file)

    return ret 
dat = open('test2.txt','w')
dat.write(convert_pdf_to_txt('./test.pdf'))
pic = 'test.jpeg'
pdf2jpg('test2.pdf',pic)