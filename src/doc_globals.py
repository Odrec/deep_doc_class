# globals.py
import os
from os.path import join, realpath, dirname, isdir
SRC_PATH = dirname(realpath(__file__))
DATA_PATH = join(join(dirname(realpath(__file__)), os.pardir), "data")
PDF_PATH = join(DATA_PATH, "pdf_files")
TXT_PATH = join(DATA_PATH, "txt_files_full")
TXT_PATH_1P = join(DATA_PATH, "txt_files_1p")
TXT_PATH_2P = join(DATA_PATH, "txt_files_2p")
TXT_PATH_5P = join(DATA_PATH, "txt_files_5p")
FEATURE_VALS_PATH = join(DATA_PATH, "feature_values")
RESULT_PATH = join(join(dirname(realpath(__file__)), os.pardir), "results")
