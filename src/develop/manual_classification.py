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

def last_clean_up(man_class_file):
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
	# input()
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

def cleaning_man_classification():
    all_powerpoint_in_presentation("manual_classes.csv")
    all_without_text_not_in_scan("clean_pres_man_class.csv")
    all_in_scan_no_text("clean_scan_man_class.csv")
    all_in_scan_not_important("clean_scan2_man_class.csv")
    new_cats("clean_scan3_man_class.csv")
    new_cat_long_official_style_mats("clean_new_cats_man_class.csv")
    clean_old_cats("clean_old_cats_man_class.csv")
    ĺast_clean_up("clean_old_cats_man_class.csv")

    args = sys.argv
    filedir = args[1]
    filedir = os.path.abspath(filedir)
    train = args[2]
    test = args[3]

    with open(train, 'r') as df:
    	reader = csv.reader(df)
    	classifications = list(reader)

    with open(test, 'r') as df:
    	reader = csv.reader(df)
    	classifications += list(reader)

    print(len(classifications))
    manual_classifiactions = {}

    fieldnames = ['doc_id','important','category']

    if(isfile("manual_classes.csv")):
    	print("Loading the current csv!")
    	with open("manual_classes.csv", "r") as csvfile:
    		reader = csv.DictReader(csvfile)
    		# next(reader, None)
    		for line in reader:
    			manual_classifiactions[line["doc_id"]]=line
    try:
    	counter = 0
    	for file,cla in classifications:
    		if(not(file in manual_classifiactions)):
    			print("Classification is: %d"%(int(float(cla)),))
    			filename = join(filedir,file+".pdf")
    			imp, cat = get_manual_classification(filename)
    			manual_classifiactions[file] = {
    				"doc_id":file,
    				"important":imp,
    				"category":cat
    				}
    		else:
    			counter += 1

    except KeyboardInterrupt as e:
    	print("Shutting down savely!")
    	raise e

    finally:
    	print("Saving csv file!")
    	with open("manual_classes.csv", "w") as csvfile:
    		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    		writer.writeheader()
    		for key, value in manual_classifiactions.items():
    			writer.writerow(value)

if __name__ == "__main__":

	false_docs = ["cc17fb8eaf055e2c8224eaaf3c10145c",
	    "43d566f90c24f874781d97905870ed23",
	    "96a92f3bcb002f22a698a62a468415e3",
	    "9c70eaa380eb60135c5f1953b1f073cf",
	    "6aaf86df732ef957de9a48e2b3c589e9",
	    "daad7c32072aaa4cdbb9ae2c02c2c9b6",
	    "e65752213cebcb6e151a815ed21512f3",
	    "5ff6d40b1dc0dde4c51a8436bc57aa20",
	    "fc8278775cf12fdf84e88493f8efc645",
	    "d047a01a190236e10d6addefe13ed3bc",
	    "7e7ff5a990f0e5756e31348ced169854",
	    "5268494d6a5b7f3bc27bc2dcbcf79748",
	    "10c8f8111c29862458987f77c05f4a39",
	    "19e69897d4842c053d781e773e5122bd",
	    "da5a198a45c690c3570a7d307597b892",
	    "f51d621abeb4c489918d904995bbcec0",
	    "ef59be13bd0d88b43fcb779912df7a66",
	    "36889863bea6feb43dbe335719aa657a",
	    "6c357bc6ddc11a3b5d199d2b1e4e6fac",
	    "2ad710735830e95c3e9ce01725552a8b",
	    "df59eaa1565f4b3d980c15a00ade3bdc",
	    "826c1f6b7d9b2bd7fc50c23a126b71a1",
	    "12d0d64889fe56c5789f8b73a8813ce9",
	    "9d2cb33abdcfefa9a56bb1f7da4ef263",
	    "a5322615e83b66d9b6c5e9a50b3eebb2",
	    "57f98112812429c28def64a14020b242",
	    "b37ce79bf563844426c7452c38564acc",
	    "00a1e2413c919d6e392442533626cb92",
	    "acda4e5d66d28ab616cd34fd09055b05",
	    "ea6c27a337aff6c262e8316984ff72e5",
	    "b2ac2fe5be5547e864f00b2d0f3d8003",
	    "7f4038da9e1fb0aa745adcf8cc8cb832",
	    "e5b0ee8a5519969ec70f5722a90a8019",
	    "a774df02a832585275de3db40f132cdd",
	    "580bebc528e28f520565e69ef6de8506",
	    "5464ee4ef790013f0caa5d16d72ae7fc",
	    "259f6473a159a782cc6f9aaf2ceda6a1",
	    "057782ecdfc1f4179d2f98c4d80da89b",
	    "06f3ad19c37c9f4460b0c6482505599b",
	    "0a32570101750613de0fc032b94f73f4",
	    "116eee145a0276280e82e0081bc39130",
	    "136d718bee74eb407355cb64f064867d",
	    "14c2d5d73ebc31122a07ab587272f82e",
	    "170754478dc23a930d22722b43dbf181",
	    "1799aad8a5a88b17cd8480fdecdd7b81",
	    "3229f885e2668f522c2d64b84a34639b",
	    "3981b966fa573e6419054a38660bbdda",
	    "44fbbe38e3512475b9e590a937c3e933",
	    "49f920671851f6a9010187bdc9743b2c",
	    "590ce47a42bf67aa1295d02e85b1a7b5",
	    "5c52e2433813f16b46264f07fed83783",
	    "5f4616a15b26f79b6190b072dba11aa8",
	    "6956ad338f080b4be0710ce32ca12976",
	    "73197946354da526a4f08756a9703ddd",
	    "74f8f36f354984572718e765d4e85189",
	    "8b9b83971a91e40e8a05bd09d0138e49",
	    "9100a9452a85c3732784074b41233a9c",
	    "a4351807fee95ad4a750546b71b657fb",
	    "ad3b3d728c8e579ef31efe61b9739c8a",
	    "ae5378982b7227176c3e42dd93c58bd4",
	    "c7cda18baa6c465a1a076bc6034b609f",
	    "d5763c8901ef9e76bbef5e673145b300",
	    "ef82cdd10e45d8184c8779361cce78ef",
	    "7daba79637e41d82c9df63465ba2f0a3",
	    "f39e2ed04d180820f05d6ff362aedd1c",
	    "9dbb018b918a13f12613a063c386e2a7",
	    "bf6a3ea6f15c2c9c96a19da3f3ba9e43",
	    "38700021b4e8b2b6f34e19c47a096980",
	    "fabe694fea65e5b15574f4eaa2a190a9",
	    "059600524e5fa4fff9c72854ae7c70dc",
	    "6fa54fffe2a2335f5927572e9757ee40",
	    "d9eebb93c161bb0131245f80e8ccc0b3",
	    "97bb3e396c0396a693f3f8cb2f3654b3",
	    "4af0c6020d07663825f905c783828379",
	    "3c16342b2169cc662f7f3c4b0e9289a1"
	]

	print(len(false_docs))
	man_class_file = "/home/kai/Workspace/deep_doc_class/deep_doc_class/data/clean_manual_class.csv"
	man_class_file = "/home/kai/Workspace/deep_doc_class/deep_doc_class/src/develop/new_clean_manual_class.csv"
	man_class = pd.read_csv(man_class_file)
	man_class = man_class.set_index("doc_id")

	filepath = "/home/kai/Workspace/deep_doc_class/deep_doc_class/data/pdf_files"
	for d_id in false_docs:
		print(man_class.loc[d_id]["class"])
		print(man_class.loc[d_id]["category"])
		print(man_class.loc[d_id]["important"])
		show_document(join(filepath,d_id+".pdf"))
		change = input("Change?")
		if(change==1 or change=="yes" or change=="y"):
			man_class.loc[d_id]["class"] = int(not(bool(man_class.loc[d_id]["class"])))

	man_class.to_csv("new_clean_manual_class.csv")
