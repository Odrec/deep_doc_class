import sys, os
from os.path import join, realpath, dirname, isdir, basename, isfile, splitext

import subprocess
import re
from PIL import Image as PI
import imageio
import matplotlib.image as mpimg
import numpy as np
from time import time

PDF_PATH = "/home/kai/Workspace/deep_doc_class/deep_doc_class/data/files_test_html"
PDFBOOK = "4d70b0a1973a94c98b50a9fd43f0ffba.pdf"
PDFPAPER = "0f6b08591d82390c4ad1a590266f92bb.pdf"
PAGELIST_BOOK = [1,2,3,4,5,6,7,8,9,10]#,11,12,13,14,15,16,17,18,19,20]
PAGELIST_PAPER = [1] # ,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

jpegDevice = "jpeg"
pngDevice = "png16m"

FNULL = open(os.devnull, 'w')

def get_pil_image_from_pdf(fp, device, file_ext, pageList=None, first_page=-1, last_page=-1):
    args = ["gs23", "-dNOPAUSE", "-dBATCH"]
    #args.append("-sPageList=" + str(pageList)[1:-1])
    args.append("-dDownScaleFactor=4")
    args.append("-sDEVICE=" + device)
    args.append("-sOutputFile=" + splitext(fp)[0] + "_%d." + file_ext)
    if(pageList):
        args.append("-sPageList=" + str(pageList)[1:-1].replace(" ", ""))
    else:
        if(not(first_page==-1)):
            args.append("-dFirstPage=%d"%(first_page,))
        if(not(last_page==-1)):
            args.append("-dLastPage=%d"%(last_page,))
    args.append(fp)
    print(args)

    #output = subprocess.Popen(args, stdout=subprocess.PIPE).communicate()[0]
    err = subprocess.Popen(args, stdout=FNULL, stderr=subprocess.STDOUT)
    # page_break_regex = b'Page [0-9]+\n'
    # page = re.split(page_break_regex, output)[-1]
    # page = page.split(b'done.\n')[-1]
    # pil_image = PI.open(io.BytesIO(page))
    # return pil_image

def clean_images (folder=PDF_PATH):
    for root, dirs, fls in os.walk(folder):
        for name in fls:
            if(not(splitext(basename(name))[1] in ['.pdf','.xml'])):
                os.remove(join(root,name))

def load_image_test(filepath):
    s1 = time()
    im = imageio.imread(filepath)
    print(im.shape)
    print(time()-s1)
    s1 = time()
    im = mpimg.imread(filepath) # best
    print(im.shape)
    print(time()-s1)
    s1 = time()
    im_frame = PI.open(filepath)
    im = np.array(im_frame.getdata())
    print(im.shape)
    print(time()-s1)


if __name__ == "__main__":
    s1 = time()
    get_pil_image_from_pdf(join(PDF_PATH,PDFBOOK), pngDevice, "png", PAGELIST_BOOK)
    print("png: " + str(time()-s1))
    # clean_images()

    s1 = time()
    get_pil_image_from_pdf(join(PDF_PATH,PDFBOOK), jpegDevice, "jpeg", PAGELIST_BOOK)
    print("jpeg: " + str(time()-s1))
    # clean_images()
    load_image_test("/home/kai/Workspace/deep_doc_class/deep_doc_class/data/files_test_html/4d70b0a1973a94c98b50a9fd43f0ffba_2.png")
