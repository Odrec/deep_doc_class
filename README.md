DeepDocClass manual and functionality

//**Any questions or inquiries please email rgaritafigue@uos.de**//

This program gives back a probability that PDF documents are copyright protected or not.
It uses structural features as well as pixel-based features processed through an ensemble of classifiers to generate a probability.

--Setting up the environment and installing libraries

1. Install a python virtual environment with python 3.5 with this command.

virtualenv -p [PATH TO PYTHON 3.5 EXECUTABLE] [PATH FOR VIRTUAL ENVIRONMENT]

For example virtualenv -p /usr/bin/python3 deepdocvirt

2. Clone the deepdocclass application from the repository.

git clone ...

3. Activate the virtual environment with the command

source deepdocvirt/bin/activate

4. Go into the project folder and install the libraries specified in the requirements.txt file with the command.

pip install -r requirements.txt

5. Before running you need to install the NLTK stopwords by going into the python shell and installing with the commands

import nltk
nltk.download('stopwords')

6. Make sure to place the pdf files you want to process on a path where you have write permissions.

--Preparing your data

This application processes any pdf file. It uses different types of features to classify the documents.
Some of those features are based on metadata information that is not available in the file so it has to be provided.
The metadata features used are the name of the folder the file was located on the server or uploaded to 
and the name of the fileon the server.
To provide this data you need a csv file with the id of the document, the folder name and the file name.
The headers of the csv file should be:

'document_id', 'filename', 'folder_name' 

To help create this file there's a helper script inside the project folder in: helper_scripts/manage_data.py

You can pass it a csv file with more columns than the three mentioned above and that contains the headers 
with the names specified above and it will create a new file named metadata_clean.csv with only three 
columns with the necessary data. For example:

python manage_data.py -meta metadata.csv

--Running the script for prediction

When you run the script the results will be saved on the project folder under results in csv and json format.
If you want to generate a report simply use the -report parameter and it will be saved also in the results directory.
If you want to choose random files for manual inspection, use the -manual parameter.

To run the script for prediction simply do the command:

For classification with basic bow and numeric features only:

python classify.pdf -fp [DOCUMENT TO CLASSIFY OR PATH OF DOCUMENTS TO CLASSIFY]

For classification including basic features and metadata features:

python classify.pdf -fp [DOCUMENT TO CLASSIFY OR PATH OF DOCUMENTS TO CLASSIFY] -meta [PATH TO METADATA CSV FILE]

For classification including deep features:

python classify.pdf -fp [DOCUMENT TO CLASSIFY OR PATH OF DOCUMENTS TO CLASSIFY] -deep

For classification including, both, deep features and metadata features:

python classify.pdf -fp [DOCUMENT TO CLASSIFY OR PATH OF DOCUMENTS TO CLASSIFY] -meta [PATH TO METADATA CSV FILE] -deep


A training section will be added shortly...
