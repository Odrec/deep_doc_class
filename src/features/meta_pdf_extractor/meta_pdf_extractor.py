__author__ = 'matsrichter'

from os.path import join, realpath, dirname, isdir, basename
MOD_PATH = dirname(realpath(__file__))
from doc_globals import*

from PyPDF2 import PdfFileReader
from pdfminer.pdfparser import PDFParser
from collections import Counter
from pdfminer.pdfdocument import PDFDocument
import numpy as np
from sklearn.externals import joblib
import csv,re

class Meta_PDF_Module:

    def __init__(self,exclude=None):
        """

        :param exclude: indexes to exclude, reduces the return data-vector by same amount.
                        1: author (positive)
                        2: creator (positive)
                        3: producer (positive)
                        4  author (negative)
                        5  creator (negative)
                        6  pruducer (negative)
        :return:
        """
        self.lib_list = self.load_lib()
        self.exclude = exclude
        self.name = ['pos_author','pos_creator','pos_producer','neg_author','neg_creator','neg_producer']
        return

    def load_lib(self):
        """
        loads 6 bow libraries
        :return: a list of 6 Libraries
        """
        try:
            with open(join(MOD_PATH,'meta_extractor_bows.txt'),'r') as save:
                lib = eval(save.read())

        except:
            return None
        return lib

    def get_function(self,fp, mp = None):
        result = np.zeros(6)
        mp = self.get_meta(fp)
        bow_list = list()
        #bow = self.get_bow(self.get_meta(fp))
        
        if isinstance(mp,dict):

            if('/Author' in mp):
                result[0], result[3] = self.get_score(self.get_bow(mp['/Author']),0)
            else:
                result[0] = np.nan 
                result[3] = 0.0
            if '/Creator' in mp:
                result[1], result[4] = self.get_score(self.get_bow(mp['/Creator']),1)
            else:
                result[1] = np.nan 
                result[4] = 0.0
            if '/Producer' in mp:
                result[2], result[5] = self.get_score(self.get_bow(mp['/Producer']),2)
            else:
                result[2] = np.nan 
                result[5] = 0.0
                
        else: result[0:5] = np.nan
            
            
        if self.exclude != None:
            r = list()
            for i in range(6):
                if i in self.exclude:
                    continue
                r.append(result[i])
                result = np.array(r)

        return result

    def get_score(self,bow,index):
        """

        :param bow: current text as bow
        :param index: 1 = author
                      2 = creator
                      3 = producer
        :return:    positive and negative score (float)
        """
        pos = 0.0
        neg = 0.0

        for key in bow:
            if key in self.lib_list[index]:
                pos += bow[key]*self.lib_list[index][key]
            if key in self.lib_list[index+3]:
                neg += bow[key]*self.lib_list[index+3][key]
                
        return pos,neg

    def sanitize(self,txt):
        txt = ''.join(i for i in txt if not str(i).isdigit())
        txt = ''.join(i for i in txt if not i == ".")
        txt = ''.join(i for i in txt if not i == ",")
        txt = ''.join(i for i in txt if not i == "-")
        txt = ''.join(i for i in txt if not i == ";")
        #txt = ''.join(i for i in txt if not i == " ")
        return txt

    def get_meta(self, fp):
        pdf_toread = PdfFileReader(fp)
        
        if not pdf_toread.isEncrypted:
            pdf_info = pdf_toread.getDocumentInfo()
        else:
            #can't read metadata because file is encrypted
            return np.nan

        return pdf_info

    def mine_meta(self,fp):
        parser = PDFParser(fp)
        doc = PDFDocument(parser)

        print(doc.info)  # The "Info" metadata
        return doc.info

    def get_bow(self,txt):
        """
        @param txt:     the pdf-content as a string
        @return:        the bow of that specific document
        """
        txt = txt.lower()
        txt = self.sanitize(txt)
        return Counter(re.findall(r'\w+', txt))

    def train(self,filenames,classes):
        pos_creators = list()
        pos_authors = list()
        pos_producer = list()
        neg_creators = list()
        neg_authors = list()
        neg_producer = list()
        for i in range(len(filenames)):
            f = filenames[i]
            if classes[i] == 'True':
                try:
                    try:
                        mp = self.get_meta(open(join(PDF_PATH,f),'rb'))
                    except:
                        continue
                    if('/Creator' in mp):
                        pos_creators.append(self.get_bow(mp['/Creator']))
                    if '/Author' in mp:
                        pos_authors.append(self.get_bow(mp['/Author']))
                    if '/Producer' in mp:
                        pos_producer.append(self.get_bow(mp['/Producer']))
                except:
                    continue
            else:
                try:
                    try:
                        mp = self.get_meta(open(join(PDF_PATH,f),'rb'))
                    except:
                        continue
                    if('/Creator' in mp):
                        neg_creators.append(self.get_bow(mp['/Creator']))
                    else:
                        neg_creators.append(Counter({'null': 1}))
                    if '/Author' in mp:
                        neg_authors.append(self.get_bow(mp['/Author']))
                    else:
                        neg_authors.append(Counter({'null': 1}))
                    if '/Producer' in mp:
                        neg_producer.append(self.get_bow(mp['/Producer']))
                    else:
                        neg_producer.append(Counter({'null': 1}))
                except:
                    continue
        #sum up
        pos_authors = sum(pos_authors,Counter())
        pos_creators = sum(pos_creators,Counter())
        pos_producer = sum(pos_producer,Counter())
        neg_authors = sum(neg_authors,Counter())
        neg_creators = sum(neg_creators,Counter())
        neg_producer = sum(neg_producer,Counter())

        lib_list = list()

        lib_list.append(self.normalize_bow(pos_authors))
        lib_list.append(self.normalize_bow(pos_creators))
        lib_list.append(self.normalize_bow(pos_producer))
        lib_list.append(self.normalize_bow(neg_authors))
        lib_list.append(self.normalize_bow(neg_creators))
        lib_list.append(self.normalize_bow(neg_producer))

        with open(join(MOD_PATH,'meta_extractor_bows.txt'),'w+') as save:
            save.write(str(lib_list))
        self.lib_list = lib_list
        return


    def normalize_bow(self,counter):
        total = 0
        for keys in counter:
            total += counter[keys]
        for key in counter:
            counter[key] /= total
        return counter




#
#filenames = list()
#file_class = list()
#a = Meta_PDF_Module()
#
##create dictionary with classifications
#with open(join(DATA_PATH,'classification.csv'),'r') as classes:
#    reader = csv.reader(classes,delimiter=';', quotechar='|')
#    for row in reader:
#        file_class.append(row[2])
#        #if(row[2] == 'True'):
#         #   continue
#        filenames.append(row[0]+'.pdf')
#
#a.train(filenames,file_class)

"""
creators = list()
authors = list()
writers = list()
producer = list()

c = 0

for f in filenames:
    try:
 #  if True:
        c += 1
        if c == 1:
            continue
       # a.mine_meta(open('./files/'+f,'rb'))
        try:
            mp = a.get_meta(open('./files/'+f,'rb'))
        except:
            continue
       # try:
        if('/Creator' in mp):
            creators.append(a.get_bow(mp['/Creator']))
        #except:
        print(c,end='\r')
     #   try:
        if '/Author' in mp:
            authors.append(a.get_bow(mp['/Author']))
     #   except:
        print(c,end='\r')
     #   try:
        if '/Producer' in mp:
            producer.append(a.get_bow(mp['/Producer']))
     #   except:
        print(c,end='\r')
    except:
        continue


s = open('sav_neg.bows.txt','w+')
s.write(str(sum(producer,Counter())))
s.write('\n\n')
s.write(str(sum(authors,Counter())))
s.write('\n\n')
s.write(str(sum(creators,Counter())))
s.write('\n'+str(c))
s.close()
"""