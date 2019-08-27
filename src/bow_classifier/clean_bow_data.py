#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 15:44:19 2018

@author: odrec
"""
import re, math, nltk
from os.path import join, dirname, abspath
from nltk.corpus import stopwords
nltk_path = join(dirname(abspath(__file__)),'nltk_data')
nltk.data.path.append(nltk_path)

def preprocess_pdf_property_string(text):
    if(text is None):
        return"None"
    else:
        text = text.lower()
        clean_text = "".join(re.findall("[a-z]{2,}",text))
        text = clean_string_regex(text, regex='[^a-z]', sub="")
        return clean_text

def preprocess_pdf_text_string(text):
    text = remove_whitespace(text)
    # words = find_regex(text)
    words = remove_stopwords(text.split())
    text =  " ".join(words)
    return text

def preprocess_pdf_metadata_string(text, lang=['german','english']):
    if(text is None or (type(text) is float and math.isnan(text))):
        return ""
    else:
        words = find_regex(text, regex=r'(?u)\b\w\w\w+\b')
        words = remove_stopwords(words)
        if 'pdf' in words: words.remove('pdf')
        return " ".join(words)

def clean_string_regex(txt, regex=';|-|\.|,|\"|[0-9]', sub=""):
    txt = txt.lower()
    txt = re.sub(regex, sub, txt)
    return txt

def remove_whitespace(txt):
    txt = re.sub("\s", " ", txt)
    return txt

def find_regex(txt, regex=r'(?u)\b\w\w\w+\b|©'):
    words = re.findall(regex,txt)
    return words

def remove_stopwords(words):
    languages = ["english", "german", "french"]
    for language in languages:
        stop_words=set(stopwords.words(language))
        words=[w for w in words if not w in stop_words]
    return words
    
def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."