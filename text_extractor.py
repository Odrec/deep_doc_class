__author__ = 'matsrichter'

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage# if the argument is set to -1, all pages are read
import bow_pdf_test
import csv
try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO
# @return a utf-8 coded string from the pdf.

def extract_text(filenames, target,source,pages = 1):
    """

    :param filenames: list of pdffiles to extract pdfs from
    :param target:    targetfolder
    :param source:    source folder
    :param pages:     number of pages per file
    :return:          True if done, else False
    """
    b = bow_pdf_test.BoW_Text_Module()
    c = 0
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



"""

filenames = list()
source = './files/'
target = './txt_files/'
#file_class = dict()

#create dictionary with classifications
with open('classification.csv','r') as classes:
    reader = csv.reader(classes,delimiter=';', quotechar='|')
    first = True
    for row in reader:
        if first:
            first = False
            continue
        filenames.append(row[0])
extract_text(filenames,target,source)

#for file in os.listdir("./files"):
#    if file.endswith(".pdf"):
#        print(file)
#        filenames.append('./files/'+file)
"""