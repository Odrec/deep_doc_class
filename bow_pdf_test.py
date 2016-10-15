__author__ = 'matsrichter'


from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
try:
    from pdfminer.pdfparser import PDFPage
except ImportError:
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

class BoW_Text_Module:

    def __init__(self, mode = 'full',text_score = None):
        """
        :param mode:
        :param text_score: a text_score-module object used for faster training, when bow is trained first
        :return:
        """
        self.error = False
        if(mode == 'reset'):
            self.lib = dict()
        else:
            self.lib = self.load_lib(mode)
        self.text_score = text_score
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
            x=1#Placeholder
            #print('troubleshooting')
            self.error = True
        return text

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
            return eval(open('bow_train_full.txt','r').read())
        else:
            return eval(open('bow_train.txt','r').read())

    def get_function(self,filepointer, metapointer = None):
        try:
            if(self.text_score == None):
                txt = self.convert_pdf_to_txt(filepointer)
            else:
                txt = self.text_score.x
        except:
            self.error = True
            return 0.0
        txt = self.sanitize(txt)
        bow = self.get_bow(txt)
        score = self.get_score(bow,self.lib)
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
        for i in range(len(filenames)):
            print(i)
            if(classes[i] == 'True'):
                continue
            with open(filenames[i],'r') as fp:
                try:
                    txt = self.convert_pdf_to_txt(fp)
                    self.text_score.len_list.append(len(txt))
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
        return

"""

#training script to create the lib
save = open('bow_train_full.txt','w')
m = BoW_Text_Module('reset')
bows = list()

filenames = list()
file_class = dict()

#create dictionary with classifications
with open('classification.csv','rb') as classes:
    reader = csv.reader(classes,delimiter=';', quotechar='|')
    first = True
    for row in reader:
        if first:
            first = False
            continue
        file_class['./files/'+row[0]+'.pdf'] = row[2]
        filenames.append("./files/"+row[0]+".pdf")

counter = 0
bag = dict()
for i in range(len(filenames)):
    counter += 1
    print(str(counter)+'/'+str(len(filenames)))
    if file_class[filenames[i]] == 'True':
        continue
    #if(counter == 100):
    #    break
    print(str(counter)+'/'+str(len(filenames)))
    fp = open(filenames[i],'r')
    try:
        b = m.get_bow(m.sanitize(m.convert_pdf_to_txt(fp)))
        bows.append(b)
    except:
        print('troubleshooting with file: '+fp.name)
    finally:
        fp.close()
    if counter%500 == 0:
        bow = sum(bows, collections.Counter())
        #bb = bow.most_common(1000)
        #bow = collections.Counter()
        #for elem in bb:
        #    bow[elem[0]] = elem[1]
        bows = [bow]
#bow = sum(bows, collections.Counter())
#bb = bow.most_common(1000)
#ow = dict()
#for elem in bb:
#    bow[elem[0]] = elem[1]

all = 0
for key in bow:
    all += bow[key]

for key in bow:
    bow[key] = float(bow[key])/float(all)

save.write(str(bow))
save.close()

"""