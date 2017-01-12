# Classification.py

import os, sys
from os.path import join, realpath, dirname, isdir, basename
# import for pathes for important folders
from doc_globals import* 

import csv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axis3d as a3d #@UnresolvedImport
from matplotlib.font_manager import FontProperties

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from scipy.stats import pearsonr, mstats

# visualization help
from prettytable import PrettyTable
import colorsys


### Neural Network
def init_keras_NN(num_input_nodes, num_hidden_nodes=100, num_hidden_layers=1):
    '''
    Creates a simple Neural Network according to the paramaters specified. There are a lot more parameters which can be chosen so its only meant to get a first impresssion of the dataset. It uses those further specifications:
        - Dense node networks on all layers
        - Sigmoid activations Functions
        - Mean Squared Error as loss function
        - Uniform initialization wheights

    @param  num_input_nodes: number of input nodes. This has to be konform with the dataset.
    @type   num_input_nodes: int

    @param  num_hidden_nodes: number of neurons in a hidden layer
    @type   num_hidden_nodes: int

    @param  num_hidden_layers: number of neuron layers between the input nodes and the final nodes
    @type   num_hidden_layers: int

    @return  model: the model of the neural net 
    @rtype   model: keras.model.Sequential
    '''
    model = Sequential()
    
    #Input layers
    model.add(Dense(num_input_nodes,input_dim=num_input_nodes, init="uniform"))
    #model.add(Activation("relu"))
    
    for i in range(0,num_hidden_layers):
        #Hidden layers
        model.add(Dense(num_hidden_nodes, init="uniform"))
        model.add(Activation("sigmoid"))
                   
    #Output layer
    #Activation function Sigmoid
    model.add(Dense(1, activation="sigmoid"))
    
    #Compile model
    #The loss function is binary_crossentropy since we are dealing with 
    #just two classes. Stochastic gradient descent
    #model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
    model.compile(loss="mean_squared_error", optimizer="sgd", metrics=["accuracy"])

    return model       

def evaluate_NN(model, train_data, train_labels, test_data, test_lables, num_epochs):
    '''
    Trains the neural network model with the training data and then tests it on the testing data.

    @param  model: A neural network model.
    @type   model: keras.model.Sequential

    @param  train_data: matrix of training data
    @type   train_data: np.array

    @param  train_labels: the classification for each training data row
    @type   train_labels: list(boolean)

    @param  test_data: matrix of testing data
    @type   test_data: np.array

    @param  test_lables: the classification for each testing data row
    @type   test_lables: list(boolean)

    @param  num_epochs: number of times that the network is trained with the whole training data 
    @type   num_epochs: int

    @return  probs: the predictions as probabilities for the testing set
    @rtype   probs: np.array
    '''
    model.fit(train_data, train_labels, nb_epoch=num_epochs, verbose=0)
    train_scores = model.evaluate(test_data, test_lables, verbose=0)
    print("train_%s: %.2f%%" % (model.metrics_names[1], train_scores[1]*100))
    probs = model.predict(test_data, verbose=0)
    return probs

def write_prediction_analysis(probs, label, doc_ids, column_names, vals, outpath):
    '''
    Gives information on all missclassified documents and those which are correct but were a close call. It seperates each case into the categories false negative, false positive (each with very high prediction rates) or close prediction (a prediction neat 0.5). For each of those cases it shows the mentioned category, the prediction probability, the true label, the document id, and all the documents feature values.

    @param  probs: the probabilities of a prediction for a dataset
    @type   probs: np.array

    @param  label: the true predictions for that dataset
    @type   label: np.array

    @param  doc_ids: the document ids that relate to the dataset
    @type   doc_ids: list(str)

    @param  column_names: the column names of each column in the dataset
    @type   column_names: list(str)

    @param  vals: the dataset itself
    @type   vals: np.array

    @param  outpath: the path to the folder where to store the file
    @type   outpath: str
    '''
    # get the the name of each row except for the feature values
    header = ["error_type", "pred", "label", "doc_id"]
    # the feature values get a number which links to the legend (otherwise the table is to huge) 
    header.extend(range(len(column_names)))

    # print the legend as 4 feature names per row
    legend = ""
    for i,name in enumerate(column_names):
        legend += str(i)+":"+"%-20s"%(name,)+"\t"
        if((i+1)%4==0):
            legend+="\n"

    # create the three tables for the respective categories
    table_hf_fp = PrettyTable(header)
    table_hf_fn = PrettyTable(header)
    table_close= PrettyTable(header)
    # for each probability decide in which category it belongs
    for i in range(len(probs)):
        # Hugh false positive error has a probability over 0.7 but is labeled as false
        if(probs[i]>0.7 and not(label[i])):
            row = ["false_pos"]
            row.append(probs[i])
            row.append(label[i])
            row.append(doc_ids[i])
            row.extend(vals[i])
            table_hf_fp.add_row(["%.3f"%(val,) if(type(val)==np.float64 or type(val)==float) else str(val) for val in row])
        # Hugh false negative error has a probability under o.3 but is labeled as true
        if(probs[i]<0.3 and label[i]):
            row = ["false_neg"]
            row.append(probs[i])
            row.append(label[i])
            row.append(doc_ids[i])
            row.extend(vals[i])
            table_hf_fn.add_row(["%.3f"%(val,) if(type(val)==np.float64 or type(val)==float) else str(val) for val in row])
        # A close call has a prediction between 0.3 and 0.7 and any label
        if(probs[i]>=0.3 and probs[i]<=0.7):
            row = []
            if(label[i]):
                if(probs[i]>=0.5):
                    row = ["correct"]
                else:
                    row = ["false_pos"]
            else:
                if(probs[i]>=0.5):
                    row = ["false_neg"]
                else:
                    row = ["correct"]
            row.append(probs[i])
            row.append(label[i])
            row.append(doc_ids[i])
            row.extend(vals[i])
            table_close.add_row(["%.3f"%(val,) if(type(val)==np.float64 or type(val)==float) else str(val) for val in row])

    # the general textwith is the width of a tabel
    text_width = len(table_hf_fp.get_string().split("\n")[0])
    # write all the tabels with headlines to a txt file
    with open(join(outpath, "high_false_preds.txt"), 'w') as f:
        f.write(("HIGH FALSE PREDICTION ERRORS\n").center(text_width))
        f.write("\n\n" + legend.center(text_width))
        f.write("\n\n" + "High False Negatives".center(text_width) + "\n")
        f.write(table_hf_fn.get_string())
        f.write("\n\n" + "High False Positives".center(text_width) + "\n")
        f.write(table_hf_fp.get_string())
        f.write("\n\n" + "Close prediction".center(text_width) + "\n")
        f.write(table_close.get_string())

def kfold_NN(model, data, labels, files, column_names, n_folds, outpath, num_epochs=100, cut=.5):
    '''
    This function performs kfold-crossvalidation over the dataset. That means that the data is splited into k parts and in ten iterations there is each time a different pat the testing set and all the remaining ones the training set. After each prediction some scores and stats are recorded and afterwards written to files.

    The first file will contain a list of tables. The first two tables contain overall scores like accuracy confusion values etc. for each iteration and their mean and std of all iterations. One is for the predictions of the testing set one for the prediction of the training set.
    The next few tables will list the different types of errors false-negative and false-positive for the training and testing set. All will contain the document_id, th true label and the documents feature values. The testing set has the predcted probaility as well. Since every document is k-1 times in the training set it gets k-1 predictions. So in the trainingstables there is the mean probability, std and the number of times the document was missclassified instead of the single probability(mean and std over only missclassified probabilities).

    The second file will contain only the missclassified documents of the testing sets and their probabilities for easier access of missclassified documents
    
    The last file will be one boxplot of the probabilities for copyright protected documents and not copyright protected ones each. Those probabilities are again only of the testing sets.

    @param  model: the model of the neural network untrained
    @type   model: keras.model.Sequential

    @param  data: the dataset
    @type   data: np.array

    @param  labels: the prediction labels for each row in the dataset
    @type   labels: np.array

    @param  files: the document ids for of the documents in the dataset
    @type   files: list(str)

    @param  column_names: the names of the columns in the dataset
    @type   column_names: list(str)

    @param  n_folds: the number of iterations/splits for the crossvalidation
    @type   n_folds: int

    @param  outpath: the path where to store the generated documents
    @type   outpath: str

    @param  num_epochs: the number of times the neural networks lerns the complete training set
    @type   num_epochs: int

    @param  cut: the probability threshold at which to decide if a prediction is positive or negative
    @type   cut: float
    '''
    # the seed makes sure to get the same random distribution every time
    seed=7
    np.random.seed(seed)
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # those are the names of the overall scores comouted during each run
    score_names = ["accuracy", "f1", "precision", "recall", "tp", "fp", "fn", "tn"]

    # there will be a table for those scores throught the runs
    test_table = PrettyTable(score_names)
    train_table = PrettyTable(score_names)
    # mean and average will be added to the table so we need all the data in a matrix
    test_mat = np.zeros((n_folds,len(score_names)))
    train_mat = np.zeros((n_folds,len(score_names)))

    # store the ids of misslassified documents in dictionaries
    train_fn_idx = {}
    train_fp_idx = {}
    test_fn_idx = {}
    test_fp_idx = {}

    # add all the probabilities to a long list to create a boxplot for a investiation of the distribution
    copyright_boxplot = []
    not_copyright_boxplot = []

    # those are list meant for only the doc_id and the probability of an error which will be written to a second document
    test_fn_rows = []
    test_fp_rows = []

    # count the itrations
    kfold_iter = 0
    # we need to reload an untrained model every iteration to make sure that the network has only the trainig set to work with
    model.save("kfold_empty.model")
    for train, test in kfold.split(data,labels):

        # split the data
        train_data = data[train]
        test_data = data[test]
        train_labels = labels[train]
        test_labels = labels[test]
        test_files = [files[i] for i in test]
        train_files = [files[i] for i in train]

        # get the untrained model again
        model = keras.models.load_model("kfold_empty.model")
        # train the model
        model.fit(train_data, train_labels, nb_epoch=num_epochs, verbose=0)

        # predict the training set
        train_prd = model.predict(train_data, verbose=0).ravel()
        print(np.shape(train_prd))
        train_prd_bin = train_prd>=cut

        # handle all the scores for the evaluation of the training set
        train_score = model.evaluate(train_data, train_labels, verbose=0)[1]*100
        train_row = [train_score]
        train_row.append(f1_score(train_labels, train_prd_bin, average="binary"))
        train_row.append(precision_score(train_labels, train_prd_bin, average="binary"))
        train_row.append(recall_score(train_labels, train_prd_bin, average="binary"))
        train_row.extend(confusion_matrix(train_labels, train_prd_bin).ravel())
        train_table.add_row(["%.3f"%(val,) if(type(val)==np.float64 or type(val)==float) else str(val) for val in train_row])
        train_mat[kfold_iter,:] = train_row

        # put the indexes of misclassifications in their respective dict
        # since every document appears k-1 times in the training set there is a list of probabilities
        for i,x in enumerate(train_prd):
            if((x >= cut) and not(train_labels[i])):
                if(train[i] in train_fp_idx):
                    train_fp_idx[train[i]].append(x)
                else:
                    train_fp_idx[train[i]] = [x]
            if((x < cut) and train_labels[i]):
                if(train[i] in train_fn_idx):
                    train_fn_idx[train[i]].append(x)
                else:
                    train_fn_idx[train[i]] = [x]

        # predict the testing set
        test_prd = model.predict(test_data, verbose=0).ravel()
        test_prd_bin = test_prd>=cut
        copyright_boxplot.extend(test_prd[test_labels==1])
        not_copyright_boxplot.extend(test_prd[test_labels==0])

        # handle all the scores for the evaluation of the testing set
        test_score = model.evaluate(test_data, test_labels, verbose=0)[1]*100
        test_row = [test_score]
        test_row.append(f1_score(test_labels, test_prd_bin, average="binary"))
        test_row.append(precision_score(test_labels, test_prd_bin, average="binary"))
        test_row.append(recall_score(test_labels, test_prd_bin, average="binary"))
        test_row.extend(confusion_matrix(test_labels, test_prd_bin).ravel())
        test_table.add_row(["%.3f"%(val,) if(type(val)==np.float64 or type(val)==float) else str(val) for val in test_row])
        test_mat[kfold_iter,:] = test_row

        # put the indexes of misclassifications in their respective dict
        # for the second document just add the document and the prob to the list of misclassifications
        # since every document appears k-1 times in the testing set there is only one prob value
        for i,x in enumerate(test_prd):
            if((x >= cut) and not(test_labels[i])):
                test_fp_idx[test[i]] = x
                test_fp_rows.append([files[test[i]],x])
            if((x < cut) and test_labels[i]):
                test_fn_idx[test[i]] = x
                test_fn_rows.append([files[test[i]],x])

        # print the accuracy after each iteration on training and test set
        print("train_accuracy: %.2f%%" % (train_score,))
        print("test_accuracy: %.2f%%" % (test_score,))

        kfold_iter+=1

    # remove the untrained model created for the run
    os.remove("kfold_empty.model")

    # the overall scores tables now get another column for mean and std deviation seperated by an empty column
    test_table.add_row(['']*len(score_names))
    train_table.add_row(['']*len(score_names))
    test_table.add_row(["%.3f"%(val,) if(type(val)==np.float64 or type(val)==float) else str(val) for val in np.mean(test_mat,axis=0)])
    test_table.add_row(["%.3f"%(val,) if(type(val)==np.float64 or type(val)==float) else str(val) for val in np.std(test_mat,axis=0)])
    train_table.add_row(["%.3f"%(val,) if(type(val)==np.float64 or type(val)==float) else str(val) for val in np.mean(train_mat,axis=0)])
    train_table.add_row(["%.3f"%(val,) if(type(val)==np.float64 or type(val)==float) else str(val) for val in np.std(train_mat,axis=0)])

    # get the the name of each row except for the feature values.
    # they are distinct for test and train because each document just appears once in the testing set so there is just one probility here. In the training set it appears k-1 times so it can be missclassified k-1 times as well. So we take the mean, std and the count of missclassifications of the sam document in this table.
    test_header = ["doc_id", "label", "prob"]
    train_header = ["doc_id", "label", "mean_probs", "std_probs", "count"]
    # the feature values get a number which links to the legend (otherwise the table is too huge) 
    test_header.extend(range(len(column_names)))
    train_header.extend(range(len(column_names)))

    # print the legend as 4 feature names per row
    legend = ""
    for i,name in enumerate(column_names):
        legend += str(i)+":"+"%-20s"%(name,)+"\t"
        if((i+1)%4==0):
            legend+="\n"

    # add the respective values to the tables
    fp_test_table = PrettyTable(test_header)
    for idx in test_fp_idx.keys():
        row = [files[idx]]
        row.append(labels[idx])
        row.append(test_fp_idx[idx])
        row.extend(data[idx,:])
        fp_test_table.add_row(["%.3f"%(val,) if(type(val)==np.float64 or type(val)==float) else str(val) for val in row])

    fn_test_table = PrettyTable(test_header)
    for idx in test_fn_idx.keys():
        row = [files[idx]]
        row.append(labels[idx])
        row.append(test_fn_idx[idx])
        row.extend(data[idx,:])
        fn_test_table.add_row(["%.3f"%(val,) if(type(val)==np.float64 or type(val)==float) else str(val) for val in row])

    fp_train_table = PrettyTable(train_header)
    for idx in train_fp_idx.keys():
        row = [files[idx]]
        row.append(labels[idx])
        row.append(np.mean(train_fp_idx[idx]))
        row.append(np.std(train_fp_idx[idx]))
        row.append(len(train_fp_idx[idx]))
        row.extend(data[idx,:])
        fp_train_table.add_row(["%.3f"%(val,) if(type(val)==np.float64 or type(val)==float) else str(val) for val in row])

    fn_train_table = PrettyTable(train_header)
    for idx in train_fn_idx.keys():
        row = [files[idx]]
        row.append(labels[idx])
        row.append(np.mean(train_fn_idx[idx]))
        row.append(np.std(train_fn_idx[idx]))
        row.append(len(train_fn_idx[idx]))
        row.extend(data[idx,:])
        fn_train_table.add_row(["%.3f"%(val,) if(type(val)==np.float64 or type(val)==float) else str(val) for val in row])

    # get the length of the different tables
    text_width_scores = len(test_table.get_string().split("\n")[0])
    text_width_test = len(fn_test_table.get_string().split("\n")[0])
    text_width_train = len(fn_train_table.get_string().split("\n")[0])

    with open(join(outpath, "cross_eval_nn.txt"), 'w') as f:
        f.write(("%d-FOLD-CROSSVALIDATION\n"%(n_folds,)).center(text_width_scores))
        f.write("\n\n" + "results - trainigset:\n".center(text_width_scores) + "\n")
        f.write(train_table.get_string())
        f.write("\n\n" + "results -testingset:\n".center(text_width_scores) + "\n")
        f.write(test_table.get_string())

        f.write("\n\n" + "testingset-details\n".center(text_width_test) + "\n\n")
        f.write(legend)
        f.write("\n\n" + "false-negatives:".center(text_width_test) + "\n")
        f.write(fn_test_table.get_string())
        f.write("\n" + "false-positives:".center(text_width_test) + "\n")
        f.write(fp_test_table.get_string())

        f.write("\n\n" + "trainingset-details\n".center(text_width_train) + "\n\n")
        f.write(legend)
        f.write("\n\n" + "false-negatives:".center(text_width_train) + "\n")
        f.write(fn_train_table.get_string())
        f.write("\n" + "false-positives:".center(text_width_train) + "\n")
        f.write(fp_train_table.get_string())

    test_fn_rows.sort(key=lambda list: list[1], reverse=False)
    test_fp_rows.sort(key=lambda list: list[1], reverse=True)

    text_width_files = len("%s\t%.3f"%(test_fn_rows[0][0],test_fn_rows[0][1]))
    with open(join(outpath,"false_pred_nn.txt"), 'w') as f:

        f.write("False Classified Documents".center(80) + "\n")
        f.write("(High percentages mean a high confidence that the document is copyright protected)".center(80) + "\n")

        f.write("\n\n" + "False Negatives".center(80))
        f.write("\n" + "(Classified as not copyright protected but they are)".center(80) + "\n\n")
        for row in test_fn_rows:
            f.write(("%s\t%.3f"%(row[0],row[1])).center(80) + "\n")

        f.write("\n\n" + "False Positives".center(80))
        f.write("\n" + "(Classified copyright protected but they are not)".center(80) + "\n\n")
        for row in test_fp_rows:
            f.write(("%s\t%.3f"%(row[0],row[1])).center(80) + "\n")

    create_boxplot([copyright_boxplot,not_copyright_boxplot], ["prd_copyright", "prd_not_copyright"], join(outpath,"log_reg_pred_boxplot.png"))


### Logistic Regression ###
def evaluate_logreg(model, train_data, train_labels, test_data, test_lables):
    '''
    Trains the logistic regression classifier with the training data and then tests it on the testing data.

    @param  model: A logistic regression model.
    @type   model: sklearn.linear_model.LogisticRegression

    @param  train_data: matrix of training data
    @type   train_data: np.array

    @param  train_labels: the classification for each training data row
    @type   train_labels: list(boolean)

    @param  test_data: matrix of testing data
    @type   test_data: np.array

    @param  test_lables: the classification for each testing data row
    @type   test_lables: list(boolean)

    @return  probs: the predictions as probabilities for the testing set
    @rtype   probs: np.array
    '''
    model.fit(train_data, train_labels)
    scores = model.score(test_data, test_lables)
    probs = model.predict_proba(test_data)[:,1]
    print("lg: %.3f%%" % (scores*100, ))
    return probs

def kfold_logreg(model, data, labels, files, column_names, n_folds, outpath, cut=.5):
    '''
    This function performs kfold-crossvalidation over the dataset. That means that the data is splited into k parts and in ten iterations there is each time a different pat the testing set and all the remaining ones the training set. After each prediction some scores and stats are recorded and afterwards written to files.

    The first file will contain a list of tables. The first two tables contain overall scores like accuracy confusion values etc. for each iteration and their mean and std of all iterations. One is for the predictions of the testing set one for the prediction of the training set.
    The next few tables will list the different types of errors false-negative and false-positive for the training and testing set. All will contain the document_id, th true label and the documents feature values. The testing set has the predcted probaility as well. Since every document is k-1 times in the training set it gets k-1 predictions. So in the trainingstables there is the mean probability, std and the number of times the document was missclassified instead of the single probability(mean and std over only missclassified probabilities).

    The second file will contain only the missclassified documents of the testing sets and their probabilities for easier access of missclassified documents
    
    The last file will be one boxplot of the probabilities for copyright protected documents and not copyright protected ones each. Those probabilities are again only of the testing sets.

    @param  model: the model of the logistic regression classifier untrained
    @type   model: keras.model.Sequential

    @param  data: the dataset
    @type   data: np.array

    @param  labels: the prediction labels for each row in the dataset
    @type   labels: np.array

    @param  files: the document ids for of the documents in the dataset
    @type   files: list(str)

    @param  column_names: the names of the columns in the dataset
    @type   column_names: list(str)

    @param  n_folds: the number of iterations/splits for the crossvalidation
    @type   n_folds: int

    @param  outpath: the path where to store the generated documents
    @type   outpath: str

    @param  cut: the probability threshold at which to decide if a prediction is positive or negative
    @type   cut: float
    '''
    
    # the seed makes sure to get the same random distribution every time
    seed=7
    np.random.seed(seed)
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # those are the names of the overall scores comouted during each run
    score_names = ["accuracy", "f1", "precision", "recall", "tp", "fp", "fn", "tn"]

    # there will be a table for those scores throught the runs
    test_table = PrettyTable(score_names)
    train_table = PrettyTable(score_names)
    # mean and average will be added to the table so we need all the data in a matrix
    test_mat = np.zeros((n_folds,len(score_names)))
    train_mat = np.zeros((n_folds,len(score_names)))

    # store the ids of misslassified documents in dictionaries
    train_fn_idx = {}
    train_fp_idx = {}
    test_fn_idx = {}
    test_fp_idx = {}

    # add all the probabilities to a long list to create a boxplot for a investiation of the distribution
    copyright_boxplot = []
    not_copyright_boxplot = []

    # those are list meant for only the doc_id and the probability of an error which will be written to a second document
    test_fn_rows = []
    test_fp_rows = []

    # count the itrations
    kfold_iter = 0

    for train, test in kfold.split(data,labels):

        # split the data
        train_data = data[train]
        test_data = data[test]
        train_labels = labels[train]
        test_labels = labels[test]
        test_files = [files[i] for i in test]
        train_files = [files[i] for i in train]

        # train the model
        model.fit(train_data, train_labels)

        # predict the training set
        train_prd = model.predict_proba(train_data)[:,1]
        train_prd_bin = train_prd>=cut

        # handle all the scores for the evaluation of the training set
        train_score = model.score(train_data, train_labels)
        train_row = [train_score]
        train_row.append(f1_score(train_labels, train_prd_bin, average="binary"))
        train_row.append(precision_score(train_labels, train_prd_bin, average="binary"))
        train_row.append(recall_score(train_labels, train_prd_bin, average="binary"))
        train_row.extend(confusion_matrix(train_labels, train_prd_bin).ravel())
        train_table.add_row(["%.3f"%(val,) if(type(val)==np.float64 or type(val)==float) else str(val) for val in train_row])
        train_mat[kfold_iter,:] = train_row

        # put the indexes of misclassifications in their respective dict
        # since every document appears k-1 times in the training set there is a list of probabilities
        for i,x in enumerate(train_prd):
            if((x >= cut) and not(train_labels[i])):
                if(train[i] in train_fp_idx):
                    train_fp_idx[train[i]].append(x)
                else:
                    train_fp_idx[train[i]] = [x]
            if((x < cut) and train_labels[i]):
                if(train[i] in train_fn_idx):
                    train_fn_idx[train[i]].append(x)
                else:
                    train_fn_idx[train[i]] = [x]

        # predict the testing set
        test_prd = model.predict_proba(test_data)[:,1]
        test_prd_bin = test_prd>=cut
        copyright_boxplot.extend(test_prd[test_labels==1])
        not_copyright_boxplot.extend(test_prd[test_labels==0])

        # handle all the scores for the evaluation of the testing set
        test_score = model.score(test_data, test_labels)
        test_row = [test_score]
        test_row.append(f1_score(test_labels, test_prd_bin, average="binary"))
        test_row.append(precision_score(test_labels, test_prd_bin, average="binary"))
        test_row.append(recall_score(test_labels, test_prd_bin, average="binary"))
        test_row.extend(confusion_matrix(test_labels, test_prd_bin).ravel())
        test_table.add_row(["%.3f"%(val,) if(type(val)==np.float64 or type(val)==float) else str(val) for val in test_row])
        test_mat[kfold_iter,:] = test_row

        # put the indexes of misclassifications in their respective dict
        # for the second document just add the document and the prob to the list of misclassifications
        # since every document appears k-1 times in the testing set there is only one prob value
        for i,x in enumerate(test_prd):
            if((x >= cut) and not(test_labels[i])):
                test_fp_idx[test[i]] = x
                test_fp_rows.append([files[test[i]],x])
            if((x < cut) and test_labels[i]):
                test_fn_idx[test[i]] = x
                test_fn_rows.append([files[test[i]],x])

        # print the accuracy after each iteration on training and test set
        print("train_accuracy: %.2f%%" % (train_score,))
        print("test_accuracy: %.2f%%" % (test_score,))

        kfold_iter+=1

    # the overall scores tables now get another column for mean and std deviation seperated by an empty column
    test_table.add_row(['']*len(score_names))
    train_table.add_row(['']*len(score_names))
    test_table.add_row(["%.3f"%(val,) if(type(val)==np.float64 or type(val)==float) else str(val) for val in np.mean(test_mat,axis=0)])
    test_table.add_row(["%.3f"%(val,) if(type(val)==np.float64 or type(val)==float) else str(val) for val in np.std(test_mat,axis=0)])
    train_table.add_row(["%.3f"%(val,) if(type(val)==np.float64 or type(val)==float) else str(val) for val in np.mean(train_mat,axis=0)])
    train_table.add_row(["%.3f"%(val,) if(type(val)==np.float64 or type(val)==float) else str(val) for val in np.std(train_mat,axis=0)])

    # get the the name of each row except for the feature values.
    # they are distinct for test and train because each document just appears once in the testing set so there is just one probility here. In the training set it appears k-1 times so it can be missclassified k-1 times as well. So we take the mean, std and the count of missclassifications of the sam document in this table.
    test_header = ["doc_id", "label", "prob"]
    train_header = ["doc_id", "label", "mean_probs", "std_probs", "count"]
    # the feature values get a number which links to the legend (otherwise the table is too huge) 
    test_header.extend(range(len(column_names)))
    train_header.extend(range(len(column_names)))

    # print the legend as 4 feature names per row
    legend = ""
    for i,name in enumerate(column_names):
        legend += str(i)+":"+"%-20s"%(name,)+"\t"
        if((i+1)%4==0):
            legend+="\n"

    # add the respective values to the tables
    fp_test_table = PrettyTable(test_header)
    for idx in test_fp_idx.keys():
        row = [files[idx]]
        row.append(labels[idx])
        row.append(test_fp_idx[idx])
        row.extend(data[idx,:])
        fp_test_table.add_row(["%.3f"%(val,) if(type(val)==np.float64 or type(val)==float) else str(val) for val in row])

    fn_test_table = PrettyTable(test_header)
    for idx in test_fn_idx.keys():
        row = [files[idx]]
        row.append(labels[idx])
        row.append(test_fn_idx[idx])
        row.extend(data[idx,:])
        fn_test_table.add_row(["%.3f"%(val,) if(type(val)==np.float64 or type(val)==float) else str(val) for val in row])

    fp_train_table = PrettyTable(train_header)
    for idx in train_fp_idx.keys():
        row = [files[idx]]
        row.append(labels[idx])
        row.append(np.mean(train_fp_idx[idx]))
        row.append(np.std(train_fp_idx[idx]))
        row.append(len(train_fp_idx[idx]))
        row.extend(data[idx,:])
        fp_train_table.add_row(["%.3f"%(val,) if(type(val)==np.float64 or type(val)==float) else str(val) for val in row])

    fn_train_table = PrettyTable(train_header)
    for idx in train_fn_idx.keys():
        row = [files[idx]]
        row.append(labels[idx])
        row.append(np.mean(train_fn_idx[idx]))
        row.append(np.std(train_fn_idx[idx]))
        row.append(len(train_fn_idx[idx]))
        row.extend(data[idx,:])
        fn_train_table.add_row(["%.3f"%(val,) if(type(val)==np.float64 or type(val)==float) else str(val) for val in row])

    # get the length of the different tables
    text_width_scores = len(test_table.get_string().split("\n")[0])
    text_width_test = len(fn_test_table.get_string().split("\n")[0])
    text_width_train = len(fn_train_table.get_string().split("\n")[0])

    with open(join(outpath, "cross_eval_log_reg.txt"), 'w') as f:
        f.write(("%d-FOLD-CROSSVALIDATION\n"%(n_folds,)).center(text_width_scores))
        f.write("\n\n" + "results - trainigset:\n".center(text_width_scores) + "\n")
        f.write(train_table.get_string())
        f.write("\n\n" + "results -testingset:\n".center(text_width_scores) + "\n")
        f.write(test_table.get_string())

        f.write("\n\n" + "testingset-details\n".center(text_width_test) + "\n\n")
        f.write(legend)
        f.write("\n\n" + "false-negatives:".center(text_width_test) + "\n")
        f.write(fn_test_table.get_string())
        f.write("\n" + "false-positives:".center(text_width_test) + "\n")
        f.write(fp_test_table.get_string())

        f.write("\n\n" + "trainingset-details\n".center(text_width_train) + "\n\n")
        f.write(legend)
        f.write("\n\n" + "false-negatives:".center(text_width_train) + "\n")
        f.write(fn_train_table.get_string())
        f.write("\n" + "false-positives:".center(text_width_train) + "\n")
        f.write(fp_train_table.get_string())

    test_fn_rows.sort(key=lambda list: list[1], reverse=False)
    test_fp_rows.sort(key=lambda list: list[1], reverse=True)

    text_width_files = len("%s\t%.3f"%(test_fn_rows[0][0],test_fn_rows[0][1]))
    with open(join(outpath,"false_pred_log_reg.txt"), 'w') as f:

        f.write("False Classified Documents".center(80) + "\n")
        f.write("(High percentages mean a high confidence that the document is copyright protected)".center(80) + "\n")

        f.write("\n\n" + "False Negatives".center(80))
        f.write("\n" + "(Classified as not copyright protected but they are)".center(80) + "\n\n")
        for row in test_fn_rows:
            f.write(("%s\t%.3f"%(row[0],row[1])).center(80) + "\n")

        f.write("\n\n" + "False Positives".center(80))
        f.write("\n" + "(Classified copyright protected but they are not)".center(80) + "\n\n")
        for row in test_fp_rows:
            f.write(("%s\t%.3f"%(row[0],row[1])).center(80) + "\n")

    create_boxplot([copyright_boxplot,not_copyright_boxplot], ["prd_copyright", "prd_not_copyright"], join(outpath,"log_reg_pred_boxplot.png"))



### Data Processing ###
def gen_train_test_split(document_ids, test_size):
    '''
    Splits the provided data into training- and testsplits

    @param  document_ids: The document ids of the classified data
    @type   document_ids: list(str)

    @param  test_size: The fraction of the data supposed to be testing-data
    @type   test_size: float

    @return  train_doc: The document ids of the training set
    @rtype   train_doc: list(str)

    @return  test_doc: The document ids of the testing set
    @rtype   test_doc: list(str)
    '''
    n_data = len(document_ids)
    n_test = int(n_data*test_size)
    doc_copy = document_ids[:]
    random.shuffle(doc_copy)
    train_doc = doc_copy[:n_test]
    test_doc = doc_copy[n_test:]

    return train_doc, test_doc

def load_data(data_file, error_feature=True, rep_nan=True, norm=True):
    '''
    Loads data from a csv file. The second last row is expected to contain the classification category and the last row the filename/document_id. 

    @param  data_file: The path to the data
    @type   data_file: str

    @param  error_feature: Flag specifying if the error feature is to be added
    @type   error_feature: boolean

    @param  rep_nan: Flag specifying if nans are to be replaced
    @type   rep_nan: boolean

    @param  norm: Flag specifying if the data s to be normalized
    @type   norm: boolean

    @return np_features: The feature value matrix from the csv file
    @rtype  np_features: np.array(float)

    @return np_classes: The classifications for the feature vectors
    @rtype  np_classes: np.array(int)

    @return filenames: The filenames/document_ids for the vectors
    @rtype  filenames: list(str)

    @return column_names: The column for the data matrix only
    @rtype  column_names: list(str)
    '''
    # load the data into a pandas dataframe
    features=pd.read_csv(data_file, header=0, delimiter=',', quoting=1, encoding='utf-8')
    
    # get the headers
    column_names = list(features)
    # get tghe data
    np_features = features.as_matrix(column_names[:-2])
    # get classification
    np_classes = np.array(features[column_names[-2]].tolist())
    # get the document_ids
    filenames = features[column_names[-1]].tolist()
    # cut away doc_id and classification column name
    column_names = column_names[:-2]

    # add an error column that adds contains a one if there is a np.nan in the a data row or a zero otherwise
    if(error_feature):
        np_features = add_error_feature(np_features)
        # add error features column name
        column_names.append("error")
    # replace nans by the columns mean
    if(rep_nan):
        np_features = replace_nan_mean(np_features)
    # normalize the data
    if(norm):
        np_features = norm_features(np_features)


    return np_features, np_classes, filenames, column_names
    
def add_error_feature(features):
    '''
    Checks for np.nan in the feature matrix rows and adds another boolean error feature

    @param  features:The feature matrix
    @type   features: np.array

    @return  features: The feature matrix with the additional error feature
    @rtype   features: np.array
    '''
    error_feature = np.zeros((len(features),1))
    
    # Go through every row of data
    for i in range(0,len(features)):
        for x in np.float64(features[i,:]):
            if(np.isnan(x)):
                # If a np.nan was found somewhere in a row add a 1 at the index and go to the next row
                error_feature[i] = 1.0
                break
    features = np.append(features, error_feature, axis=1)
    return features

def replace_nan_mean(features):
    '''
    Replaces np.nan in the feature matrix with the mean of the respective feature

    @param  features:The feature matrix
    @type   features: np.array

    @return  features: The corrected feature matrix
    @rtype   features: np.array
    '''
    # Make sure that every field in the data is of type np.float64
    features = np.array(features)
    features = features.astype(np.float64)

    # compute the mean of each column excluding the nans
    col_mean = np.nanmean(features,axis=0)
    # get positions of nans
    inds = np.where(np.isnan(features))
    # repalce them with the respective mean
    features[inds]=np.take(col_mean,inds[1])

    # # Another method to do that
    # features = np.where(np.isnan(features), np.ma.array(features, mask=np.isnan(features)).mean(axis=0), features)
    return features

def replace_nan_weighted_dist(features):
    '''
    Replaces np.nan in the feature matrix by a weighted mean of the feature. IN taking the mean of a feature the value of a row is scaled by the distance to the row where the nan is supposed to be replaced. So the mean has to be recomputed for every missing nan.

    @param  features:The feature matrix
    @type   features: np.array

    @return  features: The corrected feature matrix
    @rtype   features: np.array
    '''
    #TODO: For now just take the simply mean method
    return replace_nan_mean(features)

def norm_features(features):
    '''
    Normalize the feature matrix. Make sure to handle np.nans beforehand

    @param  features: The feature matrix
    @type   features: np.array

    @return  features: The normalized matrix
    @rtype   features: np.array
    '''
    len_feat = len(features[0])
    max_nor=np.amax(features, axis=0)
    min_nor=np.amin(features, axis=0)
    for i in range(0, len_feat):
        f_range = (max_nor[i]-min_nor[i])
        if(f_range>0):
            features[:,i] = (features[:,i]-min_nor[i])/f_range
        else:
            print("The feature at position %s has always the same value!"%(i,))
    return features

def pca(data, dims=3):
    '''
    Do PCA on the data

    @param  data: The data matrix
    @type   data: np.array

    @param  dims: The number of dimensions of the transformed matrix
    @type   dims: int

    @return  data_trans: The transformed matrix
    @rtype   data_trans: np.array

    @return  eig_pairs: The eigenvectors ordered according to their eigenvalue's magnitude
    @rtype   eig_pairs: list(tuple(float,np.array))
    '''
    (n, d) = data.shape;
    data = data - np.tile(np.mean(data, 0), (n, 1));
    cov = np.dot(data.T, data)/(n-1)

    # create a list of pair (eigenvalue, eigenvector) tuples
    eig_val, eig_vec = np.linalg.eig(cov)
    # get the sum of the eigenvalues for normalization
    sum_eig_vals = np.sum(eig_val)
    eig_pairs = []
    for x in range(0,len(eig_val)):
        eig_pairs.append((np.abs(eig_val[x])/sum_eig_vals,  np.real(eig_vec[:,x])))
    # sort the list starting with the highest eigenvalue
    eig_pairs.sort(key=lambda tup: tup[0], reverse=True)
    
    # get the transformation matrix by stacking the eigenvectors         
    M = np.hstack((eig_pairs[i][1].reshape(d,1) for i in range(0,dims)))

    # compute the transformed matrix
    data_trans = np.dot(data, M)
    return data_trans, eig_pairs

def test_bow_only(features_file):
    '''
    Test the accuracy of the bow features alone. 

    @param  features_file: The path to the feature file
    @type   features_file: str
    '''
    # load data
    f_vals, f_classes, files, f_names = load_data(features_file, False, False, False)

    # generate the probability based on the bow only
    # that means sum up values higher than 0.5 and lower than 0.5 respectively and substract those vectors
    # give the resulting vector probablities according which of the two categories occured more often
    probs = np.sum(f_vals[:,0:5]>0.5,axis=1)-np.sum(f_vals[:,0:5]<0.5,axis=1)
    probs = probs.astype(float)
    # the one side has to be at least 2 more indicators than the other side to be predicted as that side
    probs[probs>1]=1.0
    probs[probs==0]=0.5
    probs[probs==1]=0.5
    probs[probs==-1]=0.5
    probs[probs<-1]=0.0
    # assign the prediction
    preds = probs>=0.5

    # concatenate relevant values prediction and real classification vectors
    inter = np.concatenate((np.concatenate((f_vals[:,0:5],preds.reshape(len(f_vals),1)),axis=1),f_classes.reshape(len(f_vals),1)),axis=1)
    # print those rowes where a classification has been made and it was wrong
    for i in range(len(f_vals)):
        if(preds[i]!=f_classes[i] and probs[i]!=0.5):
            print(inter[i,:])

    # Get the overall scores
    score = float(np.sum([preds==f_classes]))/len(f_classes)
    print("accuracy: %.4f" %(score,))
    pos_dev = np.sum(np.abs(probs-0.5)[preds==f_classes])/np.sum([preds==f_classes])
    neg_dev = np.sum(np.abs(probs-0.5)[preds!=f_classes])/np.sum([preds!=f_classes])
    print("correct_std_0.5: %.4f" %(pos_dev,))
    print("flase_std_0.5: %.4f" %(neg_dev,))
    # Get the scores for only the classified part
    preds = probs[probs!=0.5]
    f_classes2 = f_classes[probs!=0.5]
    preds = preds>=0.5
    score2 = float(np.sum([preds==f_classes2]))/len(f_classes2)
    print("accuracy2: %.4f" %(score2,))
    print("predicted: %d/%d" %(len(f_classes2),len(f_classes)))
    print('\n')


### Visalizations ###
def scatter_3d(data, classes, filepath):
    '''
    Scatter plot 3d classification-data according to their classification

    @param  data: The data matrix
    @type   data: np.array

    @param  classes: The classification for each data row
    @type   classes: np.array

    @param  filepath: The path where to store pictures of the plot
    @type   filepath: str
    '''
    # Make sure the data s of the rght dimensions
    (n,d) = np.shape(data)
    assert d == 3 , 'Need to have 3 dimensions'
    # create a figure
    fig = plt.figure('scatter3D')
    # set the resolution to high definition
    dpi = fig.get_dpi()
    fig.set_size_inches(1920.0/float(dpi),1080.0/float(dpi))
    # create a 3d plot
    ax = fig.add_subplot(111, projection='3d')
    
    # arrange the ticks nicely
    maxTicks = 5
    loc = plt.MaxNLocator(maxTicks)
    ax.yaxis.set_major_locator(loc)
    ax.zaxis.set_major_locator(loc)
    
    # add labels
    ax.set_title('scatter3D')
    ax.set_xlabel('\n' +'Dim1', linespacing=1.5)
    ax.set_ylabel('\n' +'Dim2', linespacing=1.5)
    ax.set_zlabel('\n' +'Dim3', linespacing=1.5)
    
    # scatter the data in using different colors for each class
    t_data = data[:,:][classes==1]
    f_data = data[:,:][classes==0]
    ax.scatter(t_data[:,0],t_data[:,1],t_data[:,2],s=10, label = "protected", edgecolor=colors[0])
    ax.scatter(f_data[:,0],f_data[:,1],f_data[:,2],s=10, label = "not_protected", edgecolor=colors[1])
         
    # add legend with specific Font properties
    fontP = FontProperties()
    fontP.set_size('medium')
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0,0.7), prop=fontP)
    ax.grid('on')
    
    # set a viewing angle for the 3d picture
    ax.view_init(elev=10, azim=-90)

    # save the plot as figures from different angles
    fig.savefig(filepath+"_e10a90", bbox_extra_artists=(lgd,), bbox_inches='tight')
    ax.view_init(elev=80, azim=-90)
    fig.savefig(filepath+"_e80a-90", bbox_extra_artists=(lgd,), bbox_inches='tight')
    ax.view_init(elev=15, azim=-15)
    fig.savefig(filepath+"_e15a-15", bbox_extra_artists=(lgd,), bbox_inches='tight')

    # show the 3d plot
    plt.show()

def create_boxplot(data, collumn_names, filepath):
    '''
    Create boxplots for each data column.

    @param  data: The data matrix
    @type   data: np.array

    @param  collumn_names: The name of a column which will be the label of the according boxplot
    @type   collumn_names: list(str)

    @param  filepath: The path where to store pictures of the plot
    @type   filepath: str
    '''
    # Create a figure instance
    fig = plt.figure(1, figsize=(9, 6))
    # Create an axes instance
    ax = fig.add_subplot(111)

    ## add patch_artist=True option to ax.boxplot() 
    ## to get fill color
    bp = ax.boxplot(data, whis=[10,90], vert=False, patch_artist=True)

    ## change outline color, fill color and linewidth of the boxes
    for box in bp['boxes']:
        # change outline color
        box.set( color='#7570b3', linewidth=2)
        # change fill color
        box.set( facecolor = '#1b9e77' )
    ## change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#7570b3', linewidth=2)
    ## change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='#7570b3', linewidth=2)
    ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='#b2df8a', linewidth=2)
    ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='.', color='#e7298a', alpha=0.2)

    ## Custom x-axis labels
    ax.set_yticklabels(collumn_names)
    ## Remove top axes and right axes ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    # Save the figure
    fig.savefig(filepath, bbox_inches='tight')
    plt.close(fig)

def print_pca_eigenvectors(eig_tuples, column_names, filename):
    '''
    Prints usefull information about the eigenvectors after a pca analysis. High eigenvalues mean that the respective vector explaines a lot of the variance of the data. The numbers in a vector indicate the magnitude of the respective feature that is contributing to the vectors direction. The data needs to be normalized fot this to make any sense. 

    @param eig_pairs: The eigenvectors ordered according to their eigenvalue's magnitude
    @type  eig_pairs: list(tuple(float,np.array))

    @param  column_names: The names of a column which will be the label of the column
    @type   column_names: list(str)

    @param  filename: The path to the file where to write the analysis
    @type   filename: str
    '''
    # open the file
    with open(filename, 'w') as f:

        # create a legend of the column names
        legend = ""
        for i,name in enumerate(column_names):
            legend += "%02d"%(i+1,)+":"+"%-20s"%(name,)+"\t"
            if((i+1)%4==0):
                legend+="\n"

        # write some header
        f.write("PCA Eigenvectors".center(80) + "\n\n" + "Legend:".center(80) + "\n")
        # write the legend
        f.write(legend + "\n\n")

        # write each vector
        for row in eig_tuples:
            eig_vec = row[1]
            tmp = [(i+1,val) for (i,val) in enumerate(eig_vec)]
            tmp.sort(key=lambda tup: np.abs(tup[1]), reverse=True)  
            vec_string = ""
            for i,entry in enumerate(tmp):
                vec_string += "%02d"%(entry[0],)+" : "+"%.3f"%(entry[1],)+"\t"
                if((i+1)%4==0):
                    vec_string+="\n"
            # write the explained varaince
            f.write(("explained_variance:%.3f"%(row[0],)).center(80) + "\n")
            # write the vector values
            f.write(vec_string + "\n\n")

if __name__ == "__main__":
    args = sys.argv
    len_args = len(args)

    usage = '''Usage:
    python3 Classification.py <feature_file.csv> [<second_feature_file.csv>]
    \t- Crossvalidation if just one file,
    \t- Training on the first and evaluation on the Second for two files.

    * Make sure all files contain the same column headers and a classification column in the second last column and a document_id column as last one!!

    ** You probably want to do other stuff than the default so have a closer look at the main and the other function provided in this script...'''

    # Check for the correct amount of arguments
    if(len_args>3 or len_args<2):
        print("Error: The number of arguments is wrong!!")
        print(usage)
        sys.exit(1)

    # Load the first feature file
    features_file = args[1]
    if not os.path.isfile(features_file):
        print("Error: Features file %s doesn't exist."%(features_file,))
        exit();

    # get its name
    feature_file_name = basename(features_file).split(".")[0]
    # get the values
    f_vals, f_classes, files, f_names = load_data(features_file)

    folder = join(RESULT_PATH, feature_file_name)
    if(not(isdir(folder))):
        os.mkdir(folder)

    # # Exclude some features 
    # f_vals = f_vals[:,[0,-2]]
    # f_names = f_names[0,-2]
    # folder = join(RESULT_PATH, "exc_last_feature")

    # Initialize parameters
    hidden_layers = 1
    hidden_dims = 500
    num_epochs = 10
    conf_thresh = 0.5

    # Specify a file for evaluation statistics
    trial_name = "NN_hl"+str(hidden_layers)+"_hd"+str(hidden_dims)+"_ne"+str(num_epochs)+"_t"+str(conf_thresh)
    trial_folder = join(folder,trial_name)
    if(not(isdir(trial_folder))):
        os.mkdir(trial_folder)

    if(len_args==2):

        # # Start crossvalidation
        # print("Initiating Neural Network")
        # nn = init_keras_NN(len(f_vals[0]), hidden_dims, hidden_layers)
        # kfold_NN(nn,
        #     f_vals,
        #     f_classes,
        #     files,
        #     f_names,
        #     10,
        #     trial_folder,
        #     num_epochs,
        #     conf_thresh)
        # print("Crossvalidation done!")

        # Use logistic regression
        print("Initiating Logistic Regression")
        lg = LogisticRegression(penalty='l2', C=1, fit_intercept=True, intercept_scaling=1000)
        kfold_logreg(lg, f_vals, f_classes, files, f_names, 10, folder, conf_thresh)
        print("Crossvalidation done!")

        # # Test the bow features only
        # test_bow_only(args[1])

    else:
        # Load the testing file
        test_file = args[2]
        t_f_vals, t_f_classes, t_files, t_f_names = load_data(test_file)

        # # Exclude some features 
        # t_f_vals  = t_f_vals[:,[0,-2]]
        # t_f_names = t_f_names[0,-2]

        # # Train and test the features with the respective data
        # print("Initiating Neural Network")
        # nn = init_keras_NN(len(f_vals[0]), hidden_dims, hidden_layers)
        # evaluate_NN(nn, f_vals, f_classes, t_f_vals, t_f_classes, num_epochs)

        # Use logistic regression
        print("Initiating Logistic Regression")
        lg = LogisticRegression(penalty='l2', C=1, fit_intercept=True, intercept_scaling=1000)
        probs = evaluate_logreg(lg, f_vals, f_classes, t_f_vals, t_f_classes)
        write_prediction_analysis(probs, t_f_classes, t_files, t_f_names, t_f_vals, folder)



    # # Create boxplots for each column in the feature data
    # create_boxplot(f_vals, f_names, join(folder, "feature_box_plot_all.png"))
    # create_boxplot(f_vals[:,:][f_classes==0], f_names, join(folder, "feature_box_plot_not_copy.png"))
    # create_boxplot(f_vals[:,:][f_classes==1], f_names, join(folder, "feature_box_plot_copy.png"))
    
    # # Do PCA to have a look at the feature space
    # pca_trans, eig_tuples = pca(f_vals)
    # write_pca_eigenvectors(eig_tuples, f_names, join(folder, "eigenvectors_pca.txt"))
    # scatter_3d(pca_trans, f_classes, join(folder, "pca_3d"))



