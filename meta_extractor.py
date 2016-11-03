__author__ = 'matsrichter'
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
        self.exclude
        return

    def load_lib(self):
        """
        loads 6 bow libraries
        :return: a list of 6 Libraries
        """
        try:
            with open('meta_extractor_bows.txt','r') as save:
                lib = eval(save.read())

        except:
            return None
        return lib

    def get_function(self,fp, mp = None):
        result = np.zeros(6)
        mp = self.get_meta(fp)
        bow_list = list()
        #bow = self.get_bow(self.get_meta(fp))
        for i in range(3):
            if('/Creator' in mp and i == 0):
                result[i],result[i+3] = self.get_score(a.get_bow(mp['/Creator']),i)
            else:
                result[i] = 0.0
            if '/Author' in mp and i == 1:
               result[i],result[i+3] = self.get_score(a.get_bow(mp['/Author']),i)
            else:
                result[i] = 0.0
            if '/Producer' in mp and i == 2:
                result[i],result[i+3] = self.get_score(a.get_bow(mp['/Producer']),i)
            else:
                result[i] = 0.0
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
        pos, neg = 0.0
        for key in bow:
            if key in self.lib_list[index]:
                pos += bow[key]*self.lib_list[index][key]
            if key in self.lib_list[index]:
                neg += bow[key]*self.lib_list[index+3][key]
        return pos,neg

    def train(self,filnames,classes):
        return

    def get_meta(self, fp):
        pdf_toread = PdfFileReader(fp)
        pdf_info = pdf_toread.getDocumentInfo()
        print(str(pdf_info))
        #print((pdf_info['/CreationDate']))
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
                        mp = self.get_meta(open('./files/'+f,'rb'))
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
                        mp = self.get_meta(open('./files/'+f,'rb'))
                    except:
                        continue
                    if('/Creator' in mp):
                        neg_creators.append(self.get_bow(mp['/Creator']))
                    if '/Author' in mp:
                        neg_authors.append(self.get_bow(mp['/Author']))
                    if '/Producer' in mp:
                        neg_producer.append(self.get_bow(mp['/Producer']))
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

        with open('meta_extractor_bows.txt','w+') as save:
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



"""

filenames = list()
file_class = list()
a = Meta_PDF_Module()

#create dictionary with classifications
with open('classification.csv','r') as classes:
    reader = csv.reader(classes,delimiter=';', quotechar='|')
    for row in reader:
        file_class.append(row[2])
        #if(row[2] == 'True'):
         #   continue
        filenames.append(row[0]+'.pdf')

a.train(filenames,file_class)


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