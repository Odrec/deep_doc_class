# DeepDocClass manual and functionality

//**Any questions or inquiries please email rgaritafigue@uos.de**//

This program gives back a probability that PDF documents are copyright protected or not.
It uses structural features as well as pixel-based features processed through an ensemble of classifiers to generate a probability.

## Setting up the environment and installing libraries

**1. Install a python virtual environment with python 3.5 with this command.**

`virtualenv -p [PATH TO PYTHON 3.5 EXECUTABLE] [PATH FOR VIRTUAL ENVIRONMENT]`

For example` virtualenv -p /usr/bin/python3 deepdocvirt`

**2. Clone the deepdocclass application from the repository.**

`git clone ...`

**3. Activate the virtual environment with the command**

`source deepdocvirt/bin/activate`

**4. Go into the project folder and install the libraries specified in the requirements.txt file with the command.**

`pip install -r requirements.txt`

**5. Before running you need to install the NLTK stopwords by going into the python shell and installing with the commands**

`import nltk`
`nltk.download('stopwords')`

**6. Make sure to place the pdf files you want to process on a path where you have write permissions.**

## Preparing your data

This application processes any pdf file. It uses different types of features to classify the documents.
Some of those features are based on metadata information that is not available in the file so it has to be provided.
The metadata features used are the name of the folder the file was located on the server or uploaded to 
and the name of the fileon the server.
To provide this data you need a csv file with the id of the document, the folder name and the file name.
You can also provide the number of participants and the course name for each file if you want extra statistics 
on your report.
The headers of the csv file should be:

`'document_id', 'file_name', 'folder_name', 'number_participants', 'course_name'` 

## Running the script for prediction

When you run the script the results will be saved on the project folder under results in csv and json format.
If you want to generate a report simply use the -report parameter and it will be saved also in the results directory.
If you want to choose random files for manual inspection, use the -manual parameter.

To run the script for prediction simply do the command:

For classification with basic bow and numeric features only:

`python classify_pdf.py -fp [DOCUMENT TO CLASSIFY OR PATH OF DOCUMENTS TO CLASSIFY]`

For classification including basic features and metadata features:

`python classify_pdf.py -fp [DOCUMENT TO CLASSIFY OR PATH OF DOCUMENTS TO CLASSIFY] -meta [PATH TO METADATA CSV FILE]`

For classification including deep features:

`python classify_pdf.py -fp [DOCUMENT TO CLASSIFY OR PATH OF DOCUMENTS TO CLASSIFY] -deep`

For classification including, both, deep features and metadata features:

`python classify_pdf.py -fp [DOCUMENT TO CLASSIFY OR PATH OF DOCUMENTS TO CLASSIFY] -meta [PATH TO METADATA CSV FILE] -deep`

A training section will be added shortly...

## KMK Test

For the KMK Test the classify.run script should be used.
This script will run the files in the quantity specified by batch and will generate a merged report file
with all the statistics combined as well as a report file for a sample of 100 files. Additionally, result files for the sample
as well as all the rest of the files with a timestamp of when they were created will be created. In case of a crash or interruption, the script will resume where it left of.
NOTE: If you want to start over from scratch: delete all the files in the results directory and the file called processed_files.csv inside the data/ directory.

To use the bash script for automatizing the process of files do

`./classify.run -fp [PATH OF DOCUMENTS TO CLASSIFY] -rp [PATH FOR RESULTS] -b [BATCH QUANTITY] -meta [METADATA FILE]`

### Results and result reporting

The following results will be generated:
- A report file for a random sample of 100 files. Please send this sample report back to us as soon as it is created so we can check if the results look realistic or if something went wrong.
- A results file for all of the analyzed files. It is created incrementally over the course of approx. 2-3 days (6sec per document). 
- A folder inside the results directory with documents for the manual evaluation. You can also use the script located at help_scripts/copy_files_manual_check.run for this. To use it just run:

`./copy_files_manual_check.run -fp [PATH OF DOCUMENTS TO CLASSIFY] -rp [PATH FOR RESULTS]`

## USAGE AND PARAMETERS
```
usage: classify_pdf.py [-h] [-fp FP] [-fl FL [FL ...]] [-meta [META]] [-c C]
                       [-conf CONF] [-pf [PF]] [-po [PO]] [-train [TRAIN]]
                       [-deep [DEEP]] [-overwrite] [-report] [-manual]
                       [-sample [SAMPLE]] [-load_classified [LOAD_CLASSIFIED]]
                       [-rp [RP]] [-b B] [-t T] [-mod [MOD]]

Copyright document classification software.

required arguments:
  -fp FP                path to pdf file(s) or list of pdf files. If you use
                        saved features data this is not required, otherwise it
                        is required.

optional arguments:
  -fl FL [FL ...]       list of pdf files to process.
  -meta [META]          specifies metadata file and whether to use metadata
                        for classification.
  -c C                  specifies amount of cores for parallel processing.
  -conf CONF            specifies configuration file.
  -pf [PF]              specifies the name for the file to load the features
                        data. If -fp is also used then this flag and the data
                        to load will be ignored.
  -po [PO]              specifies that the users wants to only preprocess
                        data. The preprocess data will be saved.
  -train [TRAIN]        specifies if the user wants to train the
                        classification system and load the label data. You can
                        pass the labels file.
  -deep [DEEP]          specifies the path to the unlabeled image data needed
                        for the training procedure. If specified without a
                        path, then it is used during classification to use the
                        trained deep models.WARNING: While in training mode
                        this can take a huge amount of time and space.
  -overwrite            will overwrite all saved data, if any. If not
                        specified, the program will try to concatenate the
                        data to existing files.
  -report               Generate a report with the results and other helpful
                        statistics.
  -manual               Provides a random sample of positively classified
                        documents for manual evaluation.
  -sample [SAMPLE]      Process just a random sample of the documents. Default
                        value is 100.
  -load_classified [LOAD_CLASSIFIED]
                        Checks which files are already classified by checking
                        the file ../data/processed_files.csv and takes them
                        out of the list from files to process.
  -rp [RP]              Specifies the path to store the results, report and
                        sample for manual evaluation.
  -b B                  ONLY USED IF NOT ON TRAINING MODE. Specifies amount of
                        files per batch.
  -t T                  ONLY USED IF NOT ON TRAINING MODE. Specifies the value
                        for the threshold for the classification decision. The
                        results will be shown in the results file.
  -mod [MOD]            ONLY USED IF NOT ON TRAINING MODE. Specifies path to
                        trained models. If not specified the default path
                        (models/) will be used.

```
