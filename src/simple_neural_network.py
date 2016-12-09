# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 12:09:06 2016

Simple neural net for the prototype

@author: Renato Garita Figueiredo
"""

import sys

from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

import numpy as np
import keras
from scipy.stats import pearsonr, mstats

from prettytable import PrettyTable

class NN:
    
    def __init__(self):
        self.model = Sequential()
         
    #@params num_input_nodes Number of input nodes. In this case the amount of
    #modules for feature extraction
    def initializeNN(self, num_input_nodes, num_hidden_nodes=100, num_hidden_layers=1):
        
        print("Creating model...")
        
        #This is a well-known heuristic where the number of nodes 
        #on the hidden layer is the mean of the input and output layers
        #hidden=np.mean([num_input_nodes,1]).astype(int)
        
        #Input layers
        self.model.add(Dense(num_input_nodes,input_dim=num_input_nodes, init="uniform"))
        #self.model.add(Activation("relu"))
        
        for i in range(0,num_hidden_layers):
            #Hidden layers
            self.model.add(Dense(num_hidden_nodes, init="uniform"))
            self.model.add(Activation("sigmoid"))
                       
        #Output layer
        #Activation function Sigmoid
        self.model.add(Dense(1, activation="sigmoid"))
        
        #Compile model
        #The loss function is binary_crossentropy since we are dealing with 
        #just two classes. Stochastic gradient descent
        #self.model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
        self.model.compile(loss="mean_squared_error", optimizer="sgd", metrics=["accuracy"])        
        
        self.model.save("NN.model")        
        
        
    #@params data a list of numpy arrays. Each array is an input
    #@params labels a numpy array with the target data

    def trainNN(self, data, labels, num_epochs=100, cut=.5, k=10):
        
        seed=7
        np.random.seed(seed)
        
        cvscores = []
        f1scores = []
        prscores = []
        rcscores = []
        tnlist = []
        tplist = []
        fnlist = []
        fplist = []
        lentest = []
        lentrain = []
        lentotal = []
        #pearson = []

        
        kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

        for train, test in kfold.split(data, labels):
            train_data = data[train]
            test_data = data[test]
            train_labels = labels[train]
            test_lables = labels[test]
            #test_files = [files[i] for i in test]

            #self.model = keras.models.load_model("NN.model")

            self.model.fit(train_data, train_labels, nb_epoch=int(num_epochs), verbose=0)
            
            # evaluate the model
            scores = self.model.evaluate(test_data, test_lables, verbose=0)

            prd = self.model.predict(test_data, verbose=0)
            print("%s: %.2f%%" % (self.model.metrics_names[1], scores[1]*100))
            cvscores.append(scores[1] * 100)
            #tst = np.array([item[0] for item in data[test]])
            #pearson.append(pearsonr(tst.ravel(), labels[test].ravel()))
            for i,x in enumerate(prd):
                if x >= cut:
                    prd[i]=1.0
                else:
                    prd[i]=0.0

            f1scores.append(f1_score(test_lables, prd, average="binary"))
            prscores.append(precision_score(test_lables, prd, average="binary"))
            rcscores.append(recall_score(test_lables, prd, average="binary"))
            tn, fp, fn, tp = confusion_matrix(test_lables, prd).ravel()
            tnlist.append(tn)
            tplist.append(tp)
            fnlist.append(fn)
            fplist.append(fp)
            lentest.append(len(test))
            lentrain.append(len(train))
            lentotal.append(len(labels))

#            gnb = MultinomialNB()
#            gnb.fit(data[train],labels[train])
#            
#            print(gnb.coef_)
#            print(gnb.score(data, labels))
#            print(gnb.predict(data))
            #MetaHandler.result_html_confusiontable('conf_table.html',labels,gnb.predict(data))
        

        print("Accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
        print("F1: %.2f (+/- %.2f)" % (np.mean(f1scores), np.std(f1scores)))
        print("Precision: %.2f (+/- %.2f)" % (np.mean(prscores), np.std(prscores)))
        print("Recall: %.2f (+/- %.2f)" % (np.mean(rcscores), np.std(rcscores)))
        print("TN: %.2f (+/- %.2f)" % (np.mean(tnlist), np.std(tnlist)))
        print("TP: %.2f (+/- %.2f)" % (np.mean(tplist), np.std(tplist)))
        print("FN: %.2f (+/- %.2f)" % (np.mean(fnlist), np.std(fnlist)))
        print("FP: %.2f (+/- %.2f)" % (np.mean(fplist), np.std(fplist)))
        #print("Pearson correlation: %.2f (+/- %.2f)" % (np.mean(pearson), np.std(pearson)))
        print("TOTAL TEST: %.2f (+/- %.2f)" % (np.mean(lentest), np.std(lentest)))
        print("TOTAL TRAIN: %.2f (+/- %.2f)" % (np.mean(lentrain), np.std(lentrain)))
        print("TOTAL: %.2f (+/- %.2f)" % (np.mean(lentotal), np.std(lentotal)))

        return np.mean(cvscores), np.mean(f1scores), np.mean(prscores), np.mean(rcscores), np.mean(tnlist), np.mean(tplist), np.mean(fnlist), np.mean(fplist)

        

    def k_fold_crossvalidation(self, data, labels, files, f_names, n_folds, filepath, num_epochs=100, cut=.5):
            
            seed=7
            np.random.seed(seed)

            kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

            score_names = ["accuracy", "f1", "precision", "recall", "tp", "fp", "fn", "tn"]
            kfold_train_scores = []
            kfold_test_scores = []

            test_table = PrettyTable(score_names)
            test_mat = np.zeros((n_folds,len(score_names)))
            train_table = PrettyTable(score_names)
            train_mat = np.zeros((n_folds,len(score_names)))
            train_fn_idx = set()
            train_fp_idx = set()
            test_fn_idx = set()
            test_fp_idx = set()

            kfold_iter = 0
            for train, test in kfold.split(data,labels):

                # split the data
                train_data = data[train]
                test_data = data[test]
                train_labels = labels[train]
                test_labels = labels[test]
                test_files = [files[i] for i in test]
                train_files = [files[i] for i in train]

                # get the model
                self.model = keras.models.load_model("NN.model")
                # train the model
                self.model.fit(train_data, train_labels, nb_epoch=num_epochs, verbose=0)
                
                train_row = []
                train_scores = self.model.evaluate(train_data, train_labels, verbose=0)
                print("train_%s: %.2f%%" % (self.model.metrics_names[1], train_scores[1]*100))
                train_row.append(train_scores[1] * 100)

                train_prd = self.model.predict(train_data, verbose=0)
                train_prd_bin = np.zeros((len(train_prd),1))
                for i,x in enumerate(train_prd):
                    if x >= cut:
                        train_prd_bin[i]=1
                        if(not(train_labels[i])):
                            train_fp_idx.add(train[i])
                    else:
                        train_prd_bin[i]=0
                        if(train_labels[i]):
                            train_fn_idx.add(train[i])

                train_row.append(f1_score(train_labels, train_prd_bin, average="binary"))
                train_row.append(precision_score(train_labels, train_prd_bin, average="binary"))
                train_row.append(recall_score(train_labels, train_prd_bin, average="binary"))
                train_row.extend(confusion_matrix(train_labels, train_prd_bin).ravel())

                train_table.add_row(val_list_to_strings(train_row))
                train_mat[kfold_iter,:] = train_row

                test_row = []
                scores = self.model.evaluate(test_data, test_labels, verbose=0)
                print("test_%s: %.2f%%" % (self.model.metrics_names[1], scores[1]*100))
                test_row.append(scores[1] * 100)

                test_prd = self.model.predict(test_data, verbose=0)
                test_prd_bin = np.zeros((len(test_prd),1))
                for i,x in enumerate(test_prd):
                    if x >= cut:
                        test_prd_bin[i]=1.0
                        if(not(test_labels[i])):
                            test_fp_idx.add(test[i])
                    else:
                        test_prd_bin[i]=0.0
                        if(test_labels[i]):
                            test_fn_idx.add(test[i])

                test_row.append(f1_score(test_labels, test_prd_bin, average="binary"))
                test_row.append(precision_score(test_labels, test_prd_bin, average="binary"))
                test_row.append(recall_score(test_labels, test_prd_bin, average="binary"))
                test_row.extend(confusion_matrix(test_labels, test_prd_bin).ravel())

                test_table.add_row(val_list_to_strings(test_row))
                test_mat[kfold_iter,:] = test_row

                kfold_iter+=1
                self.model.save("NN.model")        

            test_table.add_row(['']*len(score_names))
            train_table.add_row(['']*len(score_names))

            test_table.add_row(val_list_to_strings(np.mean(test_mat,axis=0)))
            test_table.add_row(val_list_to_strings(np.std(test_mat,axis=0)))
            train_table.add_row(val_list_to_strings(np.mean(train_mat,axis=0)))
            train_table.add_row(val_list_to_strings(np.std(train_mat,axis=0)))

            header = range(0,len(f_names)+1)
            fp_test_table = PrettyTable(header)
            print(len(test_fp_idx))
            for idx in test_fp_idx:
                row = [files[idx]]
                row.extend(data[idx,:])
                fp_test_table.add_row(val_list_to_strings(row))

            fn_test_table = PrettyTable(header)
            for idx in test_fn_idx:
                row = [files[idx]]
                row.extend(data[idx,:])
                fn_test_table.add_row(val_list_to_strings(row))

            fp_train_table = PrettyTable(header)
            for idx in train_fp_idx:
                row = [files[idx]]
                row.extend(data[idx,:])
                fp_train_table.add_row(val_list_to_strings(row))

            fn_train_table = PrettyTable(header)
            for idx in train_fn_idx:
                row = [files[idx]]
                row.extend(data[idx,:])
                fn_train_table.add_row(val_list_to_strings(row))

            text_width_scores = len(test_table.get_string().split("\n")[0])
            text_width_files = len(fn_train_table.get_string().split("\n")[0])

            field_names = ""
            for i,name in enumerate(f_names):
                field_names += str(i+1)+":"+"%-20s"%(name,)+"\t"
                if((i+1)%4==0):
                    field_names+="\n"

            with open(filepath, 'w') as f:
                f.write(("%d-FOLD-CROSSVALIDATION\n"%(n_folds,)).center(text_width_scores))
                f.write("\n")
                f.write("\n")
                f.write("results - trainigset:\n".center(text_width_scores))
                f.write("\n")
                f.write(train_table.get_string())
                f.write("\n")
                f.write("\n")
                f.write("results -testingset:\n".center(text_width_scores))
                f.write("\n")
                f.write(test_table.get_string())
                f.write("\n")
                f.write("\n")
                f.write("false-negatives-details\n".center(text_width_files))
                f.write("\n")
                f.write("\n")
                f.write(field_names)
                f.write("\n")
                f.write("\n")
                f.write("trainingset:\n")
                f.write(fn_train_table.get_string())
                f.write("\n")
                f.write("testingset:\n".center(text_width_files))
                f.write(fn_test_table.get_string())
                f.write("\n")
                f.write("false-positives-details\n".center(text_width_files))
                f.write("trainingset\n".center(text_width_files))
                f.write(fp_train_table.get_string())
                f.write("\n")
                f.write("testingset\n".center(text_width_files))
                f.write(fp_test_table.get_string())

    #@params test_data a list of numpy arrays. Each array is an input
    #@params test_labels a numpy array with the target data
    def predictNN(self, data):
        self.model=keras.models.load_model("NN.model")
        prd = self.model.predict(data, verbose=0)
        return prd
        

def val_list_to_strings(vals):
    for i,val in enumerate(vals):
        if(type(val)==np.int64 or type(val)==int):
            vals[i] = str(val)
        elif(type(val)==np.float64 or type(val)==float):
            vals[i] = "%.3f"%(val,)
        else:
            vals[i] = val
    return vals
