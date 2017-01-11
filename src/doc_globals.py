# -*- coding: utf-8 -*-
import sys, os, shutil
from os.path import join, realpath, dirname, isdir
SRC_PATH = dirname(realpath(__file__))
DATA_PATH = join(join(dirname(realpath(__file__)), os.pardir), "data")
PDF_PATH = join(DATA_PATH, "pdf_files")
TXT_PATH = join(DATA_PATH, "json_txt_files")
FEATURE_VALS_PATH = join(DATA_PATH, "feature_values")
RESULT_PATH = join(join(dirname(realpath(__file__)), os.pardir), "results")
MODEL_PATH = join(DATA_PATH, "nn_models")

bcolors = {
    "HEADER" : '\033[95m',
    "OKBLUE" : '\033[94m',
    "OKGREEN" : '\033[92m',
    "WARNING" : '\033[93m',
    "FAIL" : '\033[91m',
    "ENDC" : '\033[0m',
    "BOLD" : '\033[1m',
    "UNDERLINE" : '\033[4m' 
}

def print_bcolors(formats, text):
	"""
	Add console formatting identifer to strings.

	@param formats: a list of formats for the string (has to be in the dict bcolors)
	@dtype formats: list(String)

	@param text: the string should be formatted
	@dtype text: string

	@return formated_text: the formatted string
	@dtype formated_text: string
	"""
	formated_text = ''
	for format in formats:
		formated_text += bcolors[format]
	formated_text += text + bcolors["ENDC"]
	return formated_text

def pause():
    """
    Pause the execution until Enter gets pressed
    """
    input("Press Enter to continue...")
    return