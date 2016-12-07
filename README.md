# Project Title

Pdf Copyright Classifier

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites

The program is made to work in a Unix environment, it was fully tested on Linux.

The requirements.txt file has a list of all the python 3.5 libraries you need to install for the 
code to work.

The requirements2.txt file should be run after the first requirements.txt. This is to fix a bug with scipi in some machines.

Additionally you need these external programas installed 

-ghostscript (version 9.19)

-tesseract*

-pdfinfo*

*these external programs are not used in this version of the prototype but they will possibly be used on future updates so this requirement could change.

Inputs:

-The pdf file(s)

-A metadata csv file containing the following columns (please make sure that the values on the csv have no quotes)**

>document_id = unique id of the file which should be the name of the pdf without the '.pdf' extension

>filename = the name of the file (in case the name of the file on the system is different than the id)

>folder_name = the name of the folder on the system where the file is located

>folder_description = a description of the folder where the file is located

>description = description of the file

**this prerequisite is optional since it is specific for studip platform from Osnabrück University. If you have similar metadata you can test it too but make sure to use the 
same column names

### Installing

The program itself doesn't need to be installed just copied to a local path and the script run with python 3.5.

## Running the tests

For testing the prototype just go into the src folder and type this command. The parameters on brackets [] are optional. 

```
python classify_pdf.py -d path_to_file(s) [-m path_to_metadata_csv_file] [-o name_of_output_file(s)] [-c number_of_cores] [-b number_of_files_per_batch]
```
The only parameter that is required to run is the '-d' parameter followed by the path to the pdf files or a single pdf file.

The optional '-m' parameter is followed by a csv file with the metadata for the pdf files. Take into account that the features extracted from the metadata of this file are 
specific for the metadata provided by the Osnabrück University from their studip platform. If you don't provide a metadata file then these features will all be nan values. If 
you want to provide a metada file make sure it has a format similar to the one described on the prerequisites.

The optional '-o' parameter sets the name of the output file. The default name of the output files is prediction_batch<batch_number>.csv and prediction_batch<batch_number>.json 
and can be located in the folder results/predictions. If you specify the name then the outputs will be <output_name>_batch<batch_number>.csv and similar for the json 
output. The output text files for the pdfs are extracted to data/txt_files_full. 

The optional '-c' parameter is followed by a number which indicates the number of cpu cores used to extract text from pdfs and features in parallel. If the number of cores is 
higher than the number of files in the batch ('-b' parameter) it will restrict the number of cores to the number of files per batch. If the numer of cores specified as parameter 
is higher than the number of cores available on the machine you're running the code on it will restrict the number of cores to the highest value possible on your machine. If 
this parameter is left out the default is to use just 1 core.

The optional parameter -b specifies the number of files processed per batch. The program will then generate outputs for each batch of files separetely. Use this if you have too 
many files and want partial results. If left out it will set the batch as the total number of files found on the path. 

### Notes

This prototype is to show how the overall structure of the program and the environment will look like but is not the final version so don't rely on the results of this protoype.
We will be updated with a fully working prototype shortly.

Any issues or questions regarding this source code write to rgaritafigue@uos.de 
