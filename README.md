# Project Title

Pdf Copyright Classifier

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites

The program is made to work in a Unix environment, it was fully tested on Linux.

The requirements.txt file has a list of all the python 3.5 libraries you need to install for the 
code to work.

Keras: if you have trouble installing tensorflow for keras backend you can use theano but you need to change keras 
backend since the default is tensorflow. To change it edit this file ~/.keras/keras.json.


To install the requirements.txt file follow these simple steps:

1. Install pip for python 3.5. For example in Ubuntu you can install it like this:

```
sudo apt-get install python3-pip 
```
2. Install the requirements.txt file like this:
```
pip install -r requirements.txt
```

Additionally you need these external programas installed:

-ghostscript

-tesseract*

-pdfinfo

*these external programs are not used in this version of the prototype but they will possibly be used on future updates so this requirement could change.


Required files:

-The pdf file(s)

-A output_means.csv file located in data/features_means/. This file is required for the normalization of the features

-Everything in the src folder

### Installing

The program itself needs to be copied to a local path and the script run with python 3.5.

### Usage
```
Usage: python classify_pdf.py [-fp [PATH]|[FILE]] [-conf [FILE]] [-meta [FILE] or [filename=<filename>,
folder_name=<folder_name>]] [-mod [FILE]] [-c [INT]] [-b [INT]] [-sp] [-sf] [-pf [FILE]] [-ff [FILE]] 
[-rf [FILE]] [-preprocess_only] [-features_only] [-t [FLOAT]]\n\n\
```
Arguments:

    -fp: parameter used to specify the path to the pdf file(s). This parameter is always required
    
    -conf: parameter used to pass the config file. If a config file is passed then the values specified
    in it will take precedence over the parameters given in the command line. Each parameter must be 
    specified on a new line with the name of the parameter, if the parameter has a value, the name should 
    be followed by an equal sign (=) and then the value of the parameter. Ex. metadata_file=../metadata.csv 
    or save_preprocess. If not config file is specified the default param.conf file will be used 
    Parameters that can be specified on the config file:
        metadata_file: path to metadata csv file
        batch: the quantity of files per batch
        model: path to trained model
        cores: number of cores to be used for parallel processing
        predict_threshold: the threshold used for classification of the documents
        save_preprocess: use this parameter if the preprocessing data should be saved on your hard disc
        save_features: use this parameter if the features should be saved on your hard disc
        preprocess_only: use this parameter if only the preprocessing data should be extracted and saved on
        your hard disc
        features_only: use this parameter if only the feature data should be calculated and saved on 
        your hard disc
        preprocessing_file: specifies an existing file on which to append the preprocessing data.
        features_file: specifies an existing file on which to append the feature data.
        prediction_file: specifies an existing file on which to append the result predicition data.
    
    -meta: parameter used to specify the path to the metadata csv file. It is also possible to pass the 
    metadata of a single file directly on the command line by writing filename=<filename>,
    folder_name=<folder_name> instead of the path to the metadata csv file. Be aware that if the metadata 
    is passed on the command line the -fp parameter should point to one single file and not to a path of 
    a group of files. If the metadata file is passed as paremeter only the files on it will be processed,
    any extra pdf files on the path spcified by the agument -fp that are not on the metadata file will
    be ignored.
    
    -mod: parameter used to specify the path to the trained model. If no model is specified the default 
    ones will be loaded. The default model with metadata features is NN.model, the default model without 
    metadata features is NN_noMeta.model.
    
    -c: parameter used to specify the number of cores to be used for parallel processing. 
    
    -b: parameter used to specify the number of files to be processed per batch. The preprocessing, 
    features and prediction results will be updated after each batch on the saving files.
    
    -sp: parameter used if you want to save the preprocessing data. If it doesn't exist a folder will be 
    created in '../preprocessing data'. Inside this path a 'text_files' folder will be created to store 
    the extracted text from each file and a 'features' folder will be created to store the features.
    
    -sf: parameter used if you want to save the features data.
    
    -pf: parameter used to specify the preprocessing file. If the file doesn't exist it will be created.
    The default file is 'preprocessing_data/preprocessing_data.json'. If you don't use this argument the 
    existing default file will be ovewritten.
    
    -ff: parameter used to specify the features file. If the file doesn't exist it will be created.
    The default file is 'preprocessing_data/features/features.json'. If you don't use this argument
    the existing default file will be ovewritten.
    
    -rf: parameter used to specify the result predictions file. If the file doesn't exist it will be created. 
    The default file if this parameter is not specified is '../predictions/prediction.json'. If you don't 
    use this argument the existing default file will be ovewritten.
    
    -preprocess_only: parameter used if you want to extract and save preprocessing data only.
    
    -features_only: parameter used if you want to extract and save features data only.
    
    -t: parameter used to specify the threshold for classification. The deafult value is 0.5.

### Notes

-The metadata csv (comma separated) should contain the following columns (please make sure that the values on the csv have no quotes)**

>document_id = unique id of the file which should be the name of the pdf without the '.pdf' extension

>filename = the name of the file (in case the name of the file on the system is different than the id)

>folder_name = the name of the folder on the system where the file is located

**this prerequisite is optional since it is specific for studip platform from Osnabrück University. If you have similar metadata you can test it too but make sure to use the same column names. In case you don't use the metadata a model trained without the metadata is included.

-This is a working version which will be enhanced in future versions to improve accuracy.

Any issues or questions regarding this source code write to rgaritafigue@uos.de 
