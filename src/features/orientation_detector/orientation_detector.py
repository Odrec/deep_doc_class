__author__ = 'matsrichter'
"""

This module figures out the orientation of the first page is landscape or portrait

0.0 = Landscape
1.0 = Portrait
0.5 = Error occured

"""
#from pyPdf import PdfFileReader
from PyPDF2 import PdfFileReader
import csv, numpy as np


class page_orientation_module:

    def __init__(self):
        self.error = False
        self.name = ["orientation"]
        return


    def get_function(self,fp, metapointer = None):
        try:
            pdf = PdfFileReader(fp)
            page = pdf.getPage(0).mediaBox
            if page.getUpperRight_x() - page.getLowerLeft_x() > page.getUpperRight_y() - page.getLowerRight_y():
                #Landscape
                return 0.0
            else:
                #Portrait
                return 1.0
        except:
            self.error = True
            return np.nan

    def train(self,filenames,classes,metalist = None):
        return

"""

file_class = dict()
m = page_orientation_module()

#create dictionary with classifications
with open('classification.csv','r') as classes:
    reader = csv.reader(classes,delimiter=';', quotechar='|')
    for row in reader:
        file_class['./files/'+row[0]+'.pdf'] = row[2]
p_true = 0
p_false = 0
l_true = 0
l_false = 0
landscape = 0
port = 0
true = 0
false = 0
error = 0
c = 0
for key in file_class:
    try:
        r = m.get_function(open(key,'rb'))
        c += 1
        print(c, end="\r")
        if(file_class[key] == "True"):
            true += 1
            if r == 1.0:
                p_true += 1
                port += 1
            elif r == 0.0:
                l_true += 1
                landscape += 1
            else:
                error += 1

        else:
            false += 1
            if r == 1.0:
                p_false += 1
                port += 1
            elif r == 0.0:
                l_false += 1
                landscape += 1
            else:
                error += 1

    except:
        print('trouble '+str(c))

print('Landscape True:\t'+str(l_true))
print('Landscape False:\t'+str(l_false))
print('Portrait True:\t'+str(p_true))
print('Portrait False:\t'+str(p_false))
print('Errors:\t'+str(error))
print('True:\t'+str(true))
print('False:\t'+str(false))
print('Total Landscape:.\t'+str(landscape))
print('Total Portrait:\t'+str(port))
print('Total:\t'+str(c))

"""