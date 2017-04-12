import os, sys
from os.path import basename, join, splitext, isfile, isdir
import subprocess
import json
import csv



def all_powerpoint_in_presentation(man_class_file):
    '''
	Takes the manual classified csv file and checks if every document which was created by powerpoint is in presentations

    @param  man_class_file: path to the csv with the manual classifications
    @type   man_class_file: str
    '''

def all_powerpoint_in_presentation(man_class_file):
    '''
	Takes the manual classified csv file and checks if every document which was created by powerpoint is in presentations

    @param  man_class_file: path to the csv with the manual classifications
    @type   man_class_file: str
    '''


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
