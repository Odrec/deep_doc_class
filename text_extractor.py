__author__ = 'matsrichter'

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage# if the argument is set to -1, all pages are read
import bow_pdf_test
from Text_Score_Module import TextScore
import csv
import os
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
    b = bow_pdf_test.BoW_Text_Module()
    global c
    for f in filenames:
        try:
            c += 1
            print(str(c)+'/'+str(len(filenames)),end='\r')
            fp = open(source+f+'.pdf','rb')
            tf = open(target+f+'.txt','w')
            txt = b.convert_pdf_to_txt(fp,-1)
            tf.write(txt)
        except:
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
        name = row[0]
        filenames.append(name)
        classification.append(row[2])
        if c%200 == 0:
            tl.append(filenames)
            filenames = list()
    tl.append(filenames)
thread_list = list()
global c
c = 0

for i in range(len(tl)):
    t = threading.Thread(target=extract_text,args=(tl[i],target,source,5))
    thread_list.append(t)
    t.start()
for t in tl:
    t.join()

#extract_text(filenames,target,source,5)
"""
t = TextScore(True)
b = bow_pdf_test.BoW_Text_Module(True)

print("start training...")
t.train(filenames,classification)
print("finished text score training...")
b.train(filenames,classification)
print("done.")
#extract_text(filenames,target,source)

for file in os.listdir("./files"):
    if file.endswith(".pdf"):
        #print(file)
        filenames.append(file)

"""

