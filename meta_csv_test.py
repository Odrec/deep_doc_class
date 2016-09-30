"""
meta_csv_test.py

Loads the meta data from the csv file and the classifications so you are
ready for playing around with and visualize meta data


Usage:
For playing around with the loaded data run:

Set the DOC_PATH to the provided data directory
Make sure filenames are not changed

python
execfile(<filename>)

and you are set up to play around like in matlab

@author: kai
"""
import os, sys
from os.path import join, realpath, dirname, isdir
from datetime import datetime

import csv

# https://www.decalage.info/python/html for installation
import HTML

# the module path is the path to the project folder
# beeing the parent folder of the folder of this file
# MODUL_PATH = join(dirname(realpath(__file__)), os.pardir)
MODUL_PATH = "/home/kai/Workspace/doc_class"
# this is the path to the doc_data
DOC_PATH = join(MODUL_PATH, "doc_data")

# dictionaries with fieldname and datt_idx pairs
class_fname2idx = {}
data_fname2idx = {}

def get_classifications():
	"""
	Parse the classification data

	@return classifications: the classification data
	@dtype classifications: list(list(fields))

	@return class_fnames: the fieldnames
	@dtype class_fnames: list(string)
	"""
	# open the clasification data
	data_dir = join(DOC_PATH,"classification.csv")
	class_csv = open(data_dir)
	reader = csv.DictReader(class_csv)

	# get all the fielnames and write them to the dict
	class_fnames = reader.fieldnames
	# fieldnames are all in one string seperated by semicolons (why?)
	class_fnames = class_fnames[0].split(';')
	for i in range(0,len(class_fnames)):
		class_fname2idx[class_fnames[i]] = i

	# get classifications row by row
	classifications = []
	for row in reader:
		# fields are one string seperated by semicolons (why?)
		datarow = row.values()[0].split(';')
		datarow[class_fname2idx["published"]] = datarow[class_fname2idx["published"]]=="True"
		classifications.append(datarow)



	# return classification and fieldnames
	return classifications, class_fnames

def get_classified_ids(classifications):
	"""
	Parse the classification data

	@param classifications: the classification data
	@dtype classifications: list(list(fields))

	@return ids_dict: all the ids of the classified data with row index
	@dtype ids_dict: dict(document_id:index)
	"""
	# get all the document_id's
	ids = [cf[class_fname2idx["document_id"]] for cf in classifications]
	# go through the ids and add id idx pair
	ids_dict = {}
	for i in range(len(ids)):
		ids_dict[ids[i]] = i
	return ids_dict

def get_classified_meta_data(class_data_ids):
	"""
	Get the meta data that is already classified

	@param class_data_ids: all the ids of the classified data with row index
	@dtype class_data_ids: dict(document_id:index)

	@return metadata: the already classified examples of the meta data
	@dtype metadata: list(list(fields))

	@return data_fnames: the fieldnames
	@dtype data_fnames: list(string)
	"""
	# open the meta data
	data_dir = join(DOC_PATH,"metadata.csv")
	metadata_csv = open(data_dir)
	reader = csv.DictReader(metadata_csv)

	# get the fieldnames and parse them in the index dict
	data_fnames = reader.fieldnames
	for i in range(0,len(data_fnames)):
		data_fname2idx[data_fnames[i]] = i

	# make space for meta_data
	metadata = [None]*len(class_data_ids)
	# read the data row by row
	for row in reader:
		# if it has been classified
		if(row["document_id"] in class_data_ids):
			# get the row ordered as the fieldnames
			datarow = [row[name] for name in data_fnames]
			# make ints to ints
			datarow[data_fname2idx["filesize"]] = int(datarow[data_fname2idx["filesize"]])
			# bools to bools
			datarow[data_fname2idx["author_dr"]] = bool(int(datarow[data_fname2idx["author_dr"]]))
			datarow[data_fname2idx["author_prof"]] = bool(int(datarow[data_fname2idx["author_prof"]]))
			datarow[data_fname2idx["is_pdf"]] = bool(int(datarow[data_fname2idx["is_pdf"]]))
			# and timestamps to timestampts
			datarow[data_fname2idx["upload_timestamp"]] = datetime.utcfromtimestamp(float(datarow[data_fname2idx["upload_timestamp"]]))
			# add the datarow
			metadata[class_data_ids[row["document_id"]]] = datarow

	# throw out the ids appearing in the classification set but not in the real data (why?)
	metadata = [md for md in metadata if md is not None]

	# return the data and the fieldnames
	return metadata,data_fnames

def clean_classifications(metadata, classifications):
	"""
	Throw out the ids appearing in the classification set but not in the real data (why?)

	@param classifications: the classification data
	@dtype classifications: list(list(fields)

	@param metadata: the already classified examples of the meta data
	@dtype metadata: list(list(fields))
	"""
	# make a dict from the ids appearing in the metadata
	ids = [md[data_fname2idx["document_id"]] for md in metadata]
	ids_dict =  dict.fromkeys(ids)
	# go backwards through the classification and delete ids that dont appear in the meta data
	i = len(classifications)-1
	while(i>=0):
		if(not(classifications[i][class_fname2idx["document_id"]] in ids_dict)):
			del classifications[i]
		i-=1

def get_stats(vals, op, classifications):
	"""
	Test a set of values at a choosable operator.
	Provide statistics about how the truth values of this comparison translate to the classification

	@param vals: the data examples that are to be tested
	@dtype vals: list

	@param op: the operator with which to compare the data.
	Its a binary operator with the righthand side already defined.
	@dtype op: operator

	@param classifications: the classification data
	@dtype classifications: list(list(fields)
	"""

	# get a table for the statistics
	ttable_counter = [["op/publ", "true", "false", "total"],
	["true", 0, 0, 0],
	["false", 0, 0, 0],
	["total", 0, 0, 0]]
	# go through the values and count  up the stats
	for i in range(len(vals)):
		if(op(vals[i])):
			ttable_counter[1][3]+=1
			if(classifications[i][class_fname2idx["published"]]):
				ttable_counter[3][1]+=1
				ttable_counter[1][1]+=1
			else:
				ttable_counter[3][2]+=1
				ttable_counter[1][2]+=1
		else:
			ttable_counter[2][3]+=1
			if(classifications[i][class_fname2idx["published"]]):
				ttable_counter[3][1]+=1
				ttable_counter[2][1]+=1
			else:
				ttable_counter[3][2]+=1
				ttable_counter[2][2]+=1
		ttable_counter[3][3]+=1
	# format and print each row of the table
	for row in ttable_counter:
		str_row = ""
		for cell in row:
			str_row += str(cell).center(10)
		print(str_row)


if __name__ == '__main__':


	# get the classification data
	classifications, class_fnames = get_classifications()
	# get the ids of the classifie documents
	class_data_ids = get_classified_ids(classifications)
	# get the meta data of the classified documents
	metadata, data_fnames = get_classified_meta_data(class_data_ids)
	# erase the classified ids that dont appear in the data
	clean_classifications(metadata, classifications)

	print(data_fnames)
	# example of how to use the stat evaluation
	# get interesting data set
	filesizes = [md[data_fname2idx['filesize']] for md in metadata]
	# define the requested information. The righthand side needs to be hardcoded here
	op = (lambda a: a <= 50000)
	# get the relevant stats
	get_stats(filesizes,op,classifications)


