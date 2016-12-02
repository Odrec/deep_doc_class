a__author__ = 'matsrichter'
"""
Same module as BoW_Text but only uses negative examples instead of positive

"""
from os.path import join, realpath, dirname, isdir, basename
MOD_PATH = dirname(realpath(__file__))
from doc_globals import*

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
#from pdfminer.pdfparser import PDFPage
from pdfminer.pdfpage import PDFPage
try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO
import collections, re
import numpy as np
import scipy.stats as s
import os
import csv

class Negative_BoW_Text_Module:

    def __init__(self,txt=False, mode = ''):
        self.txt = txt
        self.lib = self.load_lib(mode)
        self.path = 'txt_files_full'
        self.name = ["neg_bow_txt"]
        return

    def sanitize(self,txt):
        """
        @param txt  a page of the pdf as string
        @return     a sanitized bag of words (no blankspaces and blacklist words
                    eliminated)
        """

        #make everything lowercase
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

        # @param pages number of pages to transform to text starting from the first
    # if the argument is set to -1, all pages are read
    # @return a utf-8 coded string from the pdf.
    def convert_pdf_to_txt(self,fp,pages=-1):
        try:
            rsrcmgr = PDFResourceManager()
            retstr = StringIO()
            codec = 'utf-8'
            laparams = LAParams()
            device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
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
            #fp.close()
            device.close()
            retstr.close()
        except:
            print('troubleshooting')
            text = ''
        return text

    def convertPdf2String(fp):
      content = ""
      pdf = pyPdf.PdfFileReader(fp)
      for i in range(0, pdf.getNumPages()):
          content += pdf.getPage(i).extractText() + " \n"
          content = " ".join(content.replace(u"\xa0", u" ").strip().split())
      return content


    def get_txt(self,fp):
        path = basename(fp.name.replace('.pdf','.txt'))
        path = join(TXT_PATH, path)
        #print(path)
        fp_txt = open(path,'r')
        txt = ''
        while(True):
            try:
                tmp = fp_txt.read(1000)
                if tmp == '':
                    break
                txt += tmp
            except:
                break
        fp_txt.close()
        return txt

    def get_bow(self,txt):
        """
        @param txt:     the pdf-content as a string
        @return:        the bow of that specific document
        """
        return collections.Counter(re.findall(r'\w+', txt))

    def get_score(self,bow,lib):
        """
        :param bow:     bag of words of the current pdf
        :param lib:     the probabilities
        :return:
        """
        score = 0
        for key in bow:
            if key in lib:
                score += float(bow[key]) * float(lib[key])
        return score

    def load_lib(self,mode = 'full'):
        if(mode == 'full'):
            return eval(open(join(MOD_PATH,'bow_train_full_neg.txt'),'r').read())
        elif(mode == 'train'):
            return dict()
        else:
            return eval(open(join(MOD_PATH,'bow_train_neg.txt'),'r').read())

    def get_function(self,filepointer, metapointer = None):
        try:
            if self.txt:
                txt = self.get_txt(filepointer)
            else:
                txt = self.convert_pdf_to_txt(filepointer)
        except:
            return np.nan
        txt = self.sanitize(txt)
        bow = self.get_bow(txt)
        score = float(self.get_score(bow,self.lib))
        return score

    def train(self,filenames,classes,metalist = None):
        """
        @param filenames:   a list of paths, leading to training files
        @param classes:     a list of classifications, in the same order as the filenames
        @param metalist     not used

        This function replaced the loaded library of words and probabilities with a newly trained one, based
        on the input arguments.
        This newly generated library is NOT saved permanently and will be lost after the system terminated
        """
        lib = dict()
        all = 0
        c = 0
        for i in range(len(filenames)):
            if(classes[i] == 'False'):
                continue
            else:
                try:
                    with open(join(PDF_PATH,filenames[i]),'r') as fp:
                        print(c,end='\r')

                        if(self.txt):
                            txt = self.get_txt(fp)
                        else:
                            txt = self.convert_pdf_to_txt(fp)
                        txt = self.sanitize(txt)
                        bow = (self.get_bow(txt))
                        for key in bow:
                            all += bow[key]
                            if not key in lib:
                                lib[key] = 1
                            else:
                                lib[key] += bow[key]
                except:
                    continue
        for key in lib:
            lib[key] /= all
        self.lib = lib
        f = open(join(MOD_PATH,'bow_train_neg.txt'),'w')
        f.write(str(lib))
        f.close
        return


"""
#training script to create the lib
save = open('bow_train_neg.txt','w')
m = Negative_BoW_Text_Module(True,'train')

filenames = list()
file_class = list()

#create dictionary with classifications
with open('classification.csv','r') as classes:
    reader = csv.reader(classes,delimiter=';', quotechar='|')
    for row in reader:
        file_class.append(row[2])
        filenames.append(row[0]+'.pdf')

m.train(filenames,file_class)

"""
