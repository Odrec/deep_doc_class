__author__ = 'matsrichter'

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage# if the argument is set to -1, all pages are read
import bow_pdf_test
from Text_Score_Module import TextScore
import csv
import os
import sys
import threading
try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO
# @return a utf-8 coded string from the pdf.

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



filenames = list()
source = './files/'
target = './txt_files/'
#file_class = dict()
classification = list()
tl = list()


#create dictionary with classifications
c = 0
with open('classification.csv','r') as classes:
    reader = csv.reader(classes,delimiter=';', quotechar='|')
    first = True
    for row in reader:
        c += 1
        if first:
            first = False
            continue
        name = row[0]+'.pdf'
        filenames.append(name)
        classification.append(row[2])
       # if c%200 == 0:
            #tl.append(filenames)
           # filenames = list()
    #tl.append(filenames)
#thread_list = list()
global c
c = 0

"""
for i in range(len(tl)):
    t = threading.Thread(target=extract_text,args=(tl[i],target,source,-1))
    thread_list.append(t)
    t.start()
for t in thread_list:
    t.join()
"""
#extract_text(filenames,target,source,5)
t = TextScore(True)
b = bow_pdf_test.BoW_Text_Module(True)


print("start training...")
t.train(filenames,classification)
print("finished text score training...")
b.train(filenames,classification)
print("done.")
#extract_text(filenames,target,source)
savefile = open('save.txt','w')

t_old = TextScore()
b_old = bow_pdf_test.BoW_Text_Module()

for i in range(len(filenames)):
    try:
   # if True:
        c += 1
        print(c, end='\r')
        if c == 100:
            break
        with open('./files/'+filenames[i],'rb') as fp:
            txt_score = t.get_function(fp)
            bag_score = b.get_function(fp)
            t_old_score = t_old.get_function(fp)
            b_old_score = b_old.get_function(fp)
            savefile.write(str(txt_score)+" "+str(bag_score)+'\t'+str(t_old_score)+" "+str(b_old_score)+"\n")
    except:
        continue

savefile.close()




