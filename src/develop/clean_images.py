import os
from os.path import basename, splitext, join

doc_dir= "/home/kai/Workspace/deep_doc_class/deep_doc_class/data/pdf_files"


for root, dirs, fls in os.walk(doc_dir):
    for name in fls:
        if(not(splitext(basename(name))[1] in ['.pdf'])):

            os.remove(join(root,name))
