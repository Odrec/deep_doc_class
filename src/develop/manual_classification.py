import os, sys
from os.path import basename, join, splitext, isfile, isdir, realpath
import subprocess
import json
import csv
import pandas as pd

SRC_DIR = os.path.abspath(join(join(realpath(__file__), os.pardir),os.pardir))
sys.path.append(SRC_DIR)
sys.path.append(join(realpath(__file__), os.pardir))
from features.pdf_properties import get_pdf_properties
from features.pdf_text import get_pdf_texts_json

def ĺast_clean_up(man_class_file):
	'''
	reorganize the categories and add the real classification

	@param  man_class_file: path to the csv with the manual classifications
	@type   man_class_file: str
	'''
	row_dicts = []

	properties_path = "/home/kai/Workspace/deep_doc_class/deep_doc_class/data/pre_extracted_data/pdf_properties.json"
	datapath = "/home/kai/Workspace/deep_doc_class/deep_doc_class/data/pdf_files"
	txt_json_path = "/home/kai/Workspace/deep_doc_class/deep_doc_class/data/json_txt_files"
	train_file = "/home/kai/Workspace/deep_doc_class/deep_doc_class/data/training_data.csv"
	test_file = "/home/kai/Workspace/deep_doc_class/deep_doc_class/data/test_data.csv"
	with open(man_class_file, "r") as mc, open(train_file,"r") as train_c, open(test_file,"r") as test_c:
		train_reader = csv.reader(train_c)
		train_dict = {}
		for train_row in train_reader:
			train_dict[train_row[0]] = train_row[1]

		test_reader = csv.reader(test_c)
		test_dict = {}
		for test_row in test_reader:
			test_dict[test_row[0]] = test_row[1]
		reader = csv.DictReader(mc)
		for row in reader:
			doc_id = row["doc_id"]
			if(doc_id in train_dict):
				row["class"] = train_dict[doc_id]
			elif(doc_id in test_dict):
				row["class"] = test_dict[doc_id]
			else:
				print(doc_id)
				continue
			if(row["important"]=="1"):
				row["important"]==1
			elif(row["important"]=="0"):
				row["important"]==0
			else:
				print(doc_id)
				print("imp:" + row["important"])
				continue

			if(row["category"]=="1"):
				row["category"]=1
			elif(row["category"]=="12"):
				row["category"]=2
			elif(row["category"]=="3"):
				row["category"]=3
			elif(row["category"]=="5"):
				row["category"]=4
			elif(row["category"]=="6"):
				row["category"]=5
			elif(row["category"]=="9"):
				row["category"]=6
			elif(row["category"]=="14"):
				row["category"]=7
			elif(row["category"]=="4"):
				row["category"]=8
			elif(row["category"]=="13"):
				row["category"]=9
			elif(row["category"]=="15"):
				row["category"]=10
			elif(row["category"]=="11"):
				row["category"]=11
			elif(row["category"]=="10"):
				row["category"]=12
			elif(row["category"]=="8"):
				row["category"]=13
			else:
				print(doc_id)
				print("cat:" + row["category"])
				continue

			row_dicts.append(row)

		with open("clean_final_man_class.csv", "w") as nmc:
			writer = csv.DictWriter(nmc, fieldnames=["doc_id", "class", "important", "category"])
			writer.writeheader()
			for row in row_dicts:
				writer.writerow(row)

def clean_old_cats(man_class_file):
	'''
	clean up old unused categories 2,7 or 8

	@param  man_class_file: path to the csv with the manual classifications
	@type   man_class_file: str
	'''
	row_dicts = []

	properties_path = "/home/kai/Workspace/deep_doc_class/deep_doc_class/data/pre_extracted_data/pdf_properties.json"
	datapath = "/home/kai/Workspace/deep_doc_class/deep_doc_class/data/pdf_files"
	txt_json_path = "/home/kai/Workspace/deep_doc_class/deep_doc_class/data/json_txt_files"
	with open(man_class_file, "r") as mc:
		reader = csv.DictReader(mc)
		for row in reader:
			row_dicts.append(row)

	try:
		for i in range(len(row_dicts)):
			doc_id = row_dicts[i]["doc_id"]
			file_path = join(datapath,doc_id+".pdf")
			if(row_dicts[i]["category"]=="8"):
				print(row_dicts[i]["category"],row_dicts[i]["important"])
				imp_val, cat_val = get_manual_classification(file_path)
				row_dicts[i]["category"] = cat_val
				row_dicts[i]["important"] = imp_val
	finally:

		with open("clean_old_cats2_man_class.csv", "w") as nmc:
			writer = csv.DictWriter(nmc, fieldnames=["doc_id", "important", "category"])
			writer.writeheader()
			for row in row_dicts:
				writer.writerow(row)

def new_cat_long_official_style_mats(man_class_file):
	'''
	look in uni_materials for those ones which have offizial style (like copyright material)

	@param  man_class_file: path to the csv with the manual classifications
	@type   man_class_file: str
	'''
	row_dicts = []

	properties_path = "/home/kai/Workspace/deep_doc_class/deep_doc_class/data/pre_extracted_data/pdf_properties.json"
	datapath = "/home/kai/Workspace/deep_doc_class/deep_doc_class/data/pdf_files"
	txt_json_path = "/home/kai/Workspace/deep_doc_class/deep_doc_class/data/json_txt_files"
	with open(man_class_file, "r") as mc:
		reader = csv.DictReader(mc)
		for row in reader:
			row_dicts.append(row)

	try:
		for i in range(len(row_dicts)):
			doc_id = row_dicts[i]["doc_id"]
			file_path = join(datapath,doc_id+".pdf")
			prop_dict = get_pdf_properties(file_path, properties_path=properties_path)
			if(row_dicts[i]["category"]=="4"):
				pages = prop_dict["pages"]
				if(not(pages is None) and pages>=4):
					text = get_pdf_texts_json(doc_id, datapath, txt_json_path)
					if(type(text)==str and len(text)>7000):
						print(row_dicts[i]["category"],row_dicts[i]["important"])
						imp_val, cat_val = get_manual_classification(file_path)
						row_dicts[i]["category"] = cat_val
						row_dicts[i]["important"] = imp_val
	finally:

		with open("clean_new_cats2_man_class.csv", "w") as nmc:
			writer = csv.DictWriter(nmc, fieldnames=["doc_id", "important", "category"])
			writer.writeheader()
			for row in row_dicts:
				writer.writerow(row)

def new_cats(man_class_file):
	'''
	create new categories. presentation powerpoint and only max2pages scanned.

	@param  man_class_file: path to the csv with the manual classifications
	@type   man_class_file: str
	'''
	row_dicts = []

	properties_path = "/home/kai/Workspace/deep_doc_class/deep_doc_class/data/pre_extracted_data/pdf_properties.json"
	datapath = "/home/kai/Workspace/deep_doc_class/deep_doc_class/data/pdf_files"
	txt_json_path = "/home/kai/Workspace/deep_doc_class/deep_doc_class/data/json_txt_files"
	with open(man_class_file, "r") as mc:
		reader = csv.DictReader(mc)
		for row in reader:
			row_dicts.append(row)

	for i in range(len(row_dicts)):
		doc_id = row_dicts[i]["doc_id"]
		file_path = join(datapath,doc_id+".pdf")
		prop_dict = get_pdf_properties(file_path, properties_path=properties_path)
		if(row_dicts[i]["category"]=="1"):
			pages = prop_dict["pages"]
			if(pages<=2):
				row_dicts[i]["category"] = "12"

		elif(row_dicts[i]["category"]=="9"):
			creator = prop_dict["creator"]
			if(creator is None):
				creator = ""
			producer = prop_dict["producer"]
			if(producer is None):
				producer = ""
			if(not("powerpoint" in creator.lower()) and not("powerpoint" in producer.lower())):
				row_dicts[i]["category"] = "14"

		elif(row_dicts[i]["category"]=="4"):
			text = get_pdf_texts_json(doc_id, datapath, txt_json_path)
			if(text is None or len(text)==0):
				row_dicts[i]["category"] = "13"


	with open("clean_new_cats_man_class.csv", "w") as nmc:
		writer = csv.DictWriter(nmc, fieldnames=["doc_id", "important", "category"])
		writer.writeheader()
		for row in row_dicts:
			writer.writerow(row)

def all_in_scan_not_important(man_class_file):
	'''
	validate all scanned documents which are not important. there should be none

	@param  man_class_file: path to the csv with the manual classifications
	@type   man_class_file: str
	'''
	row_dicts = []

	properties_path = "/home/kai/Workspace/deep_doc_class/deep_doc_class/data/pre_extracted_data/pdf_properties.json"
	datapath = "/home/kai/Workspace/deep_doc_class/deep_doc_class/data/pdf_files"
	txt_json_path = "/home/kai/Workspace/deep_doc_class/deep_doc_class/data/json_txt_files"
	with open(man_class_file, "r") as mc:
		reader = csv.DictReader(mc)
		for row in reader:
			row_dicts.append(row)

	for i in range(len(row_dicts)):
		doc_id = row_dicts[i]["doc_id"]
		file_path = join(datapath,doc_id+".pdf")
		if((row_dicts[i]["category"]=="1" or row_dicts[i]["category"]=="12") and row_dicts[i]["important"]=="0"):
			print(row_dicts[i]["category"],row_dicts[i]["important"])
			imp_val, cat_val = get_manual_classification(file_path)
			row_dicts[i]["category"] = cat_val
			row_dicts[i]["important"] = imp_val

	with open("clean_scan3_man_class.csv", "w") as nmc:
		writer = csv.DictWriter(nmc, fieldnames=["doc_id", "important", "category"])
		writer.writeheader()
		for row in row_dicts:
			writer.writerow(row)

def all_in_scan_no_text(man_class_file):
	'''
	check scanned category for files which have text extracted

	@param  man_class_file: path to the csv with the manual classifications
	@type   man_class_file: str
	'''
	row_dicts = []

	properties_path = "/home/kai/Workspace/deep_doc_class/deep_doc_class/data/pre_extracted_data/pdf_properties.json"
	datapath = "/home/kai/Workspace/deep_doc_class/deep_doc_class/data/pdf_files"
	txt_json_path = "/home/kai/Workspace/deep_doc_class/deep_doc_class/data/json_txt_files"
	with open(man_class_file, "r") as mc:
		reader = csv.DictReader(mc)
		for row in reader:
			row_dicts.append(row)

	for i in range(len(row_dicts)):
		doc_id = row_dicts[i]["doc_id"]
		file_path = join(datapath,doc_id+".pdf")
		text = get_pdf_texts_json(doc_id, datapath, txt_json_path)
		if(row_dicts[i]["category"]=="1"):
			if(not(text is None) and len(text)>0):
				print(row_dicts[i]["category"], row_dicts[i]["important"])
				imp_val, cat_val = get_manual_classification(file_path)
				row_dicts[i]["category"] = cat_val
				row_dicts[i]["important"] = imp_val

	with open("clean_scan2_man_class.csv", "w") as nmc:
		writer = csv.DictWriter(nmc, fieldnames=["doc_id", "important", "category"])
		writer.writeheader()
		for row in row_dicts:
			writer.writerow(row)

def all_without_text_not_in_scan(man_class_file):
	'''
	check all files which have no text if they are in scanned

	@param  man_class_file: path to the csv with the manual classifications
	@type   man_class_file: str
	'''
	row_dicts = []

	properties_path = "/home/kai/Workspace/deep_doc_class/deep_doc_class/data/pre_extracted_data/pdf_properties.json"
	datapath = "/home/kai/Workspace/deep_doc_class/deep_doc_class/data/pdf_files"
	txt_json_path = "/home/kai/Workspace/deep_doc_class/deep_doc_class/data/json_txt_files"
	with open(man_class_file, "r") as mc:
		reader = csv.DictReader(mc)
		for row in reader:
			row_dicts.append(row)

	for i in range(len(row_dicts)):
		if(i%100==0):
			print(i)
		doc_id = row_dicts[i]["doc_id"]
		file_path = join(datapath,doc_id+".pdf")
		prop_dict = get_pdf_properties(file_path, properties_path=properties_path)
		creator = prop_dict["creator"]
		text = get_pdf_texts_json(doc_id, datapath, txt_json_path)
		cat = row_dicts[i]["category"]
		imp = row_dicts[i]["important"]
		if(text is None or len(text)==0):
			if(creator is None or not("powerpoint" in creator.lower())):
				if(not(cat=="1") or not(imp=="1")):
					if(cat=="9" or cat=="11"):
						continue

					print(row_dicts[i]["category"], row_dicts[i]["important"])
					imp_val, cat_val = get_manual_classification(file_path)
					row_dicts[i]["category"] = cat_val
					row_dicts[i]["important"] = imp_val

	with open("clean_scan_man_class.csv", "w") as nmc:
		writer = csv.DictWriter(nmc, fieldnames=["doc_id", "important", "category"])
		writer.writeheader()
		for row in row_dicts:
			writer.writerow(row)

def all_powerpoint_in_presentation(man_class_file):
	'''
	check that all documents generated by powerpoint are in powerpoint

	@param  man_class_file: path to the csv with the manual classifications
	@type   man_class_file: str

	'''
	row_dicts = []

	properties_path = "/home/kai/Workspace/deep_doc_class/deep_doc_class/data/pre_extracted_data/pdf_properties.json"
	datapath = "/home/kai/Workspace/deep_doc_class/deep_doc_class/data/pdf_files"
	with open(man_class_file, "r") as mc:
		reader = csv.DictReader(mc)
		for row in reader:
			row_dicts.append(row)

	for i in range(len(row_dicts)):
		doc_id = row_dicts[i]["doc_id"]
		file_path = join(datapath,doc_id+".pdf")
		prop_dict = get_pdf_properties(file_path, properties_path=properties_path)
		creator = prop_dict["creator"]
		if(not(creator is None) and "powerpoint" in creator.lower()):
			if(not(row_dicts[i]["category"]=="9")):
				imp_val, cat_val = get_manual_classification(file_path)
				row_dicts[i]["category"] = cat_val
				row_dicts[i]["important"] = imp_val

	with open("clean_pres_man_class.csv", "w") as nmc:
		writer = csv.DictWriter(nmc, fieldnames=["doc_id", "important", "category"])
		writer.writeheader()
		for row in row_dicts:
			writer.writerow(row)

def show_document(filename):
	args = ["evince", "--fullscreen", filename]
	plot = subprocess.Popen(args, stdout=subprocess.PIPE,
			stderr=subprocess.PIPE)
	input()
	return


def get_manual_classification(filename):
	'''
	Opens a pdf document and asks for a manual classification.

    @param  filename: path to a pdf document
    @type   filename: str

    @return  imp_val: flag indication if the pdf is an important one
    @rtype   imp_val: int

    @return  cat_val: number assigning the documnet to a category
    @rtype   cat_val: int
    '''
	args = ["evince", "--fullscreen", filename]
	plot = subprocess.Popen(args, stdout=subprocess.PIPE,
			stderr=subprocess.PIPE)

	correct = False
	while(not(correct)):
		imp_val = None
		while(imp_val is None):
			imp = input("Important?\n")
			if(imp=='1'):
				imp_val = 1
				print("IMP=1")
			elif(imp=='0'):
				imp_val = 0
				print("IMP=0")
			else:
				print("Important needs to be 0 or 1!")

		cat_val = None
		while(cat_val is None):
			cat = input("Category?\n")
			try:
				cat_val=int(cat)
				print("CAT=%d"%(cat_val,))
			except ValueError:
				cat_val = None
				print("Category needs to be 0 or 1!")
				continue

		cor = input("Correct?\n")
		if(len(cor)==0):
			correct=True

	return imp_val, cat_val



if __name__ == "__main__":
	# all_powerpoint_in_presentation("manual_classes.csv")
	# all_without_text_not_in_scan("clean_pres_man_class.csv")
	# all_in_scan_no_text("clean_scan_man_class.csv")
	# all_in_scan_not_important("clean_scan2_man_class.csv")
	# new_cats("clean_scan3_man_class.csv")
	# new_cat_long_official_style_mats("clean_new_cats_man_class.csv")
	# clean_old_cats("clean_old_cats_man_class.csv")
	# ĺast_clean_up("clean_old_cats_man_class.csv")

	# args = sys.argv
	# filedir = args[1]
	# filedir = os.path.abspath(filedir)
	# train = args[2]
	# test = args[3]
	#
	# with open(train, 'r') as df:
	# 	reader = csv.reader(df)
	# 	classifications = list(reader)
	#
	# with open(test, 'r') as df:
	# 	reader = csv.reader(df)
	# 	classifications += list(reader)
	#
	# print(len(classifications))
	# manual_classifiactions = {}
	#
	# fieldnames = ['doc_id','important','category']
	#
	# if(isfile("manual_classes.csv")):
	# 	print("Loading the current csv!")
	# 	with open("manual_classes.csv", "r") as csvfile:
	# 		reader = csv.DictReader(csvfile)
	# 		# next(reader, None)
	# 		for line in reader:
	# 			manual_classifiactions[line["doc_id"]]=line
	# try:
	# 	counter = 0
	# 	for file,cla in classifications:
	# 		if(not(file in manual_classifiactions)):
	# 			print("Classification is: %d"%(int(float(cla)),))
	# 			filename = join(filedir,file+".pdf")
	# 			imp, cat = get_manual_classification(filename)
	# 			manual_classifiactions[file] = {
	# 				"doc_id":file,
	# 				"important":imp,
	# 				"category":cat
	# 				}
	# 		else:
	# 			counter += 1
	#
	# except KeyboardInterrupt as e:
	# 	print("Shutting down savely!")
	# 	raise e
	#
	# finally:
	# 	print("Saving csv file!")
	# 	with open("manual_classes.csv", "w") as csvfile:
	# 		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
	# 		writer.writeheader()
	# 		for key, value in manual_classifiactions.items():
	# 			writer.writerow(value)

	doc_ids = [
		"c0c19a68d3e54e98dbbe839f1da12a48",
		"b70b5e44a2b2cb750092b63165c2a585",
		"136427ec5c8f990c9a38c7b197b26791",
		"de75eb121afef673bce3ae37116953ee",
		"823b9cea3ff46d7c7e2f6d40d1300a44",
		"556a8ab7e42c60d02c0fb783b9c5dcb3",
		"b881e48f5d59d74468bd52dc8c137ec3",
		"61d8ae283d943968c006e725cff315f5",
		"d1be25e9fe083d925b901322f58cce50",
		"abf19f5ebf048dbd98b2e592acbe332f",
		"925f3d0e3d12f9b07197ceda6c8376db",
		"e5178ff03bb21e40281c083021dd5d80",
		"f2a37c75153d9f6820d055c9e783128b",
		"465d36857466cc8b8461b7aad694466f",
		"48cbd1c6ed5fc9438f2721a9edf1744b",
		"a460fb0ad26247740124c7799932f977",
		"6b5e9e9ce2e29547726ae8f8abd2ca82",
		"86752187f9b846281f2fdca9312be147",
		"3d22da7e7491e5b71708b915c162e164",
		"2e1371cfcd980e966336f9b8e5c868e4",
		"41f18feced9305d0331fe8c8b53b6d11",
		"5ea39e1ba08fb05acc9e15221fea2431",
		"46fe2f38e7e10be55cecc7e87599e88a",
		"578b8421d18d1dd4f45af6b10447ea18",
		"f97fb3eb831fe7256e752234a9f95244",
		"4ddbb6ea38241fe8e862ac4ecafbe7e2",
		"6bed7318ce7d2aee785fed66e156aad7",
		"8385edc77c0c2642eeac51aaa814a2d3",
		"56c7aa5427fa26527b5073f2aff932c6",
		"aa38e362278f6624c9bebce96446361f",
		"6ebc8ee8fd87580ad497e0702cd490f5",
		"6332c56cd4403f39f4a8859f4cac44f1",
		"cecf4da70a8b544235be830e00f8e080",
		"126d11d8d35f88639f61e353c2ceeee5",
		"07592ec3c73fa533d09fc94d79e323cd",
		"6d77166becb337665a0b3b52edd57419"
	]	

	print(len(doc_ids))
	man_class_file = "/home/kai/Workspace/deep_doc_class/deep_doc_class/data/cleaned_manual_class.csv"
	man_class = pd.read_csv(man_class_file)
	man_class = man_class.set_index("doc_id")

	filepath = "/home/kai/Workspace/deep_doc_class/deep_doc_class/data/pdf_files"
	for d_id in doc_ids:
		print(man_class.loc[d_id]["category"])
		print(man_class.loc[d_id]["important"])
		show_document(join(filepath,d_id+".pdf"))
