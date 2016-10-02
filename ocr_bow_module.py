# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 16:18:12 2016

OCR test script

@author: Mats Leon Richter
"""

from wand.image import Image
from PIL import Image as PI
import pyocr
import pyocr.builders
import io

import os, os.path, sys
import csv
import numpy as np

class OCR_BoW_Module:
    
    def __init__(self):
        
        #LOAD THE LIBRARRY HERE
        self.library = self.get_library()
        return
    
    def get_library(self):
        return dict()
    
    def get_text(self,fp):
        """
        @param      fp the filepointer
        @return     list of strings containing all words inside the pdf
        
        this function extracts data from a pdf using ocr
        """
        
        #get handle of ocr lib
        tool = pyocr.get_available_tools()[0]
        lang = tool.get_available_languages()[1]
        #setup lists
        req_image = list()
        final_text = list()
        
        try:
            image_pdf = Image(filename=fp.name, resolution=300)
            image_jpeg = image_pdf.convert('jpeg')
        
        except:
            print('Error opening file: '+fp.name)
            return []
            
        for img in image_jpeg.sequence:
            img_page = Image(image=img)
            req_image.append(img_page.make_blob('jpeg'))
        
        for img in req_image:
            # redo
            txt = tool.image_to_string(PI.open(io.BytesIO(img)),
                                       lang=lang,
                                       builder=pyocr.builders.TextBuilder())                          
            final_text.append(txt)
            
        return txt
        
    def create_bag(self,txt,bow=dict()):
        """
        @param txt  a list of strings (one string per page)
        @return     a dictionarry with a keys beeing 
        """
        for t in txt:
            page = t.split()
            for word in page:
                if word in bow:
                    bow[word] += 1
                else:
                    bow[word] = 1
        return bow
    
    def sanitize(self,txt):
        """
        @param txt  a page of the pdf as string
        @return     a sanitized bag of words (no blankspaces and blacklist words 
                    eliminated)
        """
        #convert everything to lowercase
        txt = txt.lower()

        #get rid of some non-word symbols
        txt = txt.replace(',','')
        txt = txt.replace('.','')
        txt = txt.replace('?','')
        txt = txt.replace('!','')
        txt = txt.replace('\(','')
        txt = txt.replace('\)','')
        txt = txt.replace('\[','')
        txt = txt.replace('\]','')
        txt = txt.replace('\n',' ')
        txt = txt.replace(';','')
        
        #get rid of common words with noe meaning on their own
        txt = txt.replace(' and ','')
        txt = txt.replace(' und ','')
        txt = txt.replace(' der ','')
        txt = txt.replace(' die ','')
        txt = txt.replace(' das ','')
        txt = txt.replace('dass','')
        txt = txt.replace(' that ','')
        txt = txt.replace(' die ','')
        txt = txt.replace(' be ','')
        txt = txt.replace(' to ','')
        txt = txt.replace(' a ','')
        txt = txt.replace(' of ','')
        txt = txt.replace(' ist ','')
        txt = txt.replace(' is ','')
        
        return txt
    
    def get_score(self,bow):
        """
        @param      bow  the bag of words
        @return     the score of the given file
        
        """
        score = 0.0      
        for key, value in bow:
            if key in self.library:
                score += self.librrary[key]*value
        return score
        
    def get_function(self,filepointer, metapointer='None'):
        """
        @param filepointer  a filepointer to a pdf-file
        @param metapointer  not used
        @return             a 'copyright-score' represented by a single float64 value        
        
        This function returns a 'copyright-score' which is loosely tight to
        a bag-of-words bayesian classification using ocr
        of a pdf given by the filepointer
        """
        #get string
        txt = get_text(filepointer)
        for t in range(len(txt)):
            txt[t] = self.sanitize(txt[t])
        bow = create_bag(txt)
        
        return get_score(bow)


############
#TESTSCRIPT#
############

filenames = list()
file_class = dict()
bow = OCR_BoW_Module()

#create dictionary with classifications
with open('classification.csv','rb') as classes:
    reader = csv.reader(classes,delimiter=';', quotechar='|')
    for row in reader:
        file_class['./files/'+row[0]+'.pdf'] = row[2]

for file in os.listdir("./files"):
    if file.endswith(".pdf"):
#        print(file)
        filenames.append('./files/'+file)
        
counter = 0
bag = dict()
for i in range(len(filenames)):
    if file_class[filenames[i]] == 'True':
        continue
    if(counter == 100):
        break
    counter += 1
    print(str(counter)+'/'+str(len(filenames)))
    fp = open(filenames[i],'r')
    txt = str(bow.get_text(fp))
    txt = bow.sanitize(txt)
    fp.close()
    bag = bow.create_bag(txt,bag)
    #except:
     #   print("Error opening file: "+filenames[i])

c = 0    
save = open('lib','w')
for key in bag:
    c += bag[key]
for key in bag:
    bag[key] = float(bag[key])/float(c)
save.write(bag)
save.close()
    