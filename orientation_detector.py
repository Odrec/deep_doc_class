__author__ = 'matsrichter'
"""

This module figures out the orientation of the first page is landscape or portrait

0.0 = Landscape
1.0 = Portrait
0.5 = Error occured

"""
#from pyPdf import PdfFileReader
from PyPDF2 import PdfFileReader


class page_orientation_module:

    def __init__(self):
        self.error = False
        return


    def get_function(self,fp, metapointer = None):
        try:
            pdf = PdfFileReader(open('test2.pdf','rb'))
            page = pdf.getPage(0).mediaBox
            if page.getUpperRight_x() - page.getUpperLeft_x() > page.getUpperRight_y() - page.getLowerRight_y():
                return 0.0
            else:
                return 1.0
        except:
            self.error = True
            return 0.5

    def train(self,filenames,classes,metalist = None):
        return


