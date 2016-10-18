__author__ = 'matsrichter'

"""
This scanner detector uses a textline handed over by the BoW module for text
extraction or extracts the text by itself and checks if the given string has length of one.
If so, the file is most likely a scan. When this is not the case.
"""

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import csv
try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO
import os

class Naive_Scan_Detector:

    def __init__(self,txt_module = None):
        """

        :param txt_module: a text-score module, this module must be called first
        :return:
        """
        self.error = False
        self.text_module = txt_module
        return

    # @param pages number of pages to transform to text starting from the first
    # if the argument is set to -1, all pages are read
    # @return a utf-8 coded string from the pdf.
    def convert_pdf_to_txt(self,fp,pages=-1):
        try:
            rsrcmgr = PDFResourceManager()
            retstr = StringIO()
            codec = 'utf-8'
            laparams = LAParams()
            device = TextConverter(rsrcmgr, retstr, laparams=laparams)
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
            self.error = True
            return 666 #because errors are evil
            #print('troubleshooting')
        return len(text)

    def get_function(self,filepointer, metapointer = None):

        if(self.text_module == None):
            result = self.convert_pdf_to_txt(filepointer)
        else:
            result = len(self.text_module.x)
        #print(result)

        try:
            if(result > 1):
                return 0.0
            elif self.error:
                return 0.5
            else:
                #print("HIT")
                return 1.0
        except:
            self.error = True
            return 0.5

    def train(self,filenames,classes,metalist = None):
         return


#training script to create the lib
m = Naive_Scan_Detector()

filenames = list()
file_class = dict()

#create dictionary with classifications
with open('classification.csv','r') as classes:
    reader = csv.reader(classes,delimiter=';', quotechar='|')
    for row in reader:
        file_class['./files/'+row[0]+'.pdf'] = row[2]

for file in os.listdir("./files"):
    if file.endswith(".pdf"):
#        print(file)
        filenames.append('./files/'+file)

counter = 0
true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0
inter = 0
for i in range(len(filenames)):
    counter += 1
    print(str(counter)+'/'+str(len(filenames)),end='\r')
    try:
        fp = open(filenames[i],'rb')
        r = m.get_function(fp)
        if file_class[filenames[i]] == 'True':
            if r == 1.0:
                true_pos += 1
            elif r == 0.0:
                false_neg += 1
            else:
                inter += 1
        else:
            if r == 1.0:
                false_pos += 1
            elif r == 0.0:
                true_neg += 1
            else:
                inter += 1
    except:
        inter += 1

print("True Positive Rate:\t"+str(float(true_pos)/float(counter)))
print("True Negative Rate:\t"+str(float(true_neg)/float(counter)))
print("False Positive Rate:\t"+str(float(false_pos)/float(counter)))
print("False Negative Rate:\t"+str(float(false_neg)/float(counter)))


print("\nTrue Positive:\t"+str((true_pos)))
print("True Negative:\t"+str((true_neg)))
print("False Positive:\t"+str((false_pos)))
print("False Negative:\t"+str((false_neg)))



