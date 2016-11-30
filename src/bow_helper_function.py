# bow_helper_function.py
import sys
sys.path.append('/usr/lib/python3.5/lib-dynload')

from os.path import join, realpath, dirname, isdir, basename
MOD_PATH = dirname(realpath(__file__))

import nltk
from nltk.corpus import stopwords
nltk.data.path.append(join(MOD_PATH,'nltk_data'))  # setting path to files

from sklearn.feature_extraction.text import CountVectorizer
from langdetect import detect_langs
from langdetect import detect
import re
import treetaggerwrapper as ttwp
from collections import Counter

def get_lang(txt, get_prob=False):
	if(get_prob):
		langs = detect_langs(txt)
		language = langs[0]
		return language.lang, language.prob
	else:
		language = detect(txt)
		return language

def remove_stopwords(words, langs=['english']):
	usage = "words should be a dict<str,_> or list\nlangs are: english, german, french"
	languages = ["english", "german", "french"]
	if(type(words) == list):
		for language in langs:
			if(language not in languages):
				print("Wrong language!")
				print(usage)
				sys.exit(1)
			stop_words=set(stopwords.words(language))
			words=[w for w in words if not w in stop_words]
		return words

	elif(type(words) == dict):
		d_keys = list(words)
		for language in langs:
			stop_words=set(stopwords.words(language))
			if(len(stop_words)<len(d_keys)):
				for s_word in stop_words:
					if(s_word in words):
						del words[s_word]
			else:
				for k_word in d_keys:
					if(k_word in stop_words):
						del words[k_word]
		return words
	print("Not the right type of words")
	print(usage)
	sys.exit(1)

def keep_letters_and_numbers(txt):
	txt=txt.lower()
	txt=remove_newline(txt)
	german = '\u00fc\u00e4\u00f6\u00df'
	french = '\u00e0\u00e1\u00e2\u00e3\u00e6\u00e7\u00e8\u00e9\u00ea\u00ec\u00ed\u00ee\u00f1\u00f2\u00f3\u00f4\u00f5\u00f9\u00fa\u00fb'
	re_txt=re.sub(u'[^a-z0-9\u0020\u00fc\u00e4\u00f6\u00df\u00e0\u00e1\u00e2\u00e3\u00e6\u00e7\u00e8\u00e9\u00ea\u00ec\u00ed\u00ee\u00f1\u00f2\u00f3\u00f4\u00f5\u00f9\u00fa\u00fb]', "", txt)
	return re_txt

def remove_newline(txt):
	txt = txt.replace('\n',' ')
	return txt

def remove_single_chars(words):
	for word in words:
		if(len(word)<2):
			words.remove(word)
	return words

def lemmatizer(txt, taggerdir, taggerlang):
	tt = ttwp.TreeTagger(TAGDIR=taggerdir, TAGLANG=taggerlang)
	taglist = tt.tag_text(txt)
	lemmalist = []
	for tag in taglist:
		lemmalist.append(tag.split('\t')[2])
	return lemmalist

def get_word_count(words):
	counts = Counter(words)
	return dict(counts)

if __name__ == "__main__":
	text = "Das ist ein Testtext g, um zu zeigen, was die einzelnen Funktionen g können oder nicht könnt.\nDeshalb sind auch Feler999 eingebaut."
	lang = get_lang(text)
	print(lang)
	lang,prob = get_lang(text, True)
	print(lang)
	print(prob)
	text = keep_letters_and_numbers(text)
	print(text)
	word_list = lemmatizer(text, "/home/kai/opt/treewrapper", "de")
	print(word_list)
	word_list = remove_single_chars(word_list)
	print(word_list)
	word_list = get_word_count(word_list)
	print(word_list)
	word_list = remove_stopwords(word_list, langs=['german'])
	print(word_list)






