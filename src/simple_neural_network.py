# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 12:09:06 2016

Simple neural net for the prototype

@author: Renato Garita Figueiredo
"""

from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

import numpy as np
import keras
from scipy.stats import pearsonr, mstats

class NN:
    
    def __init__(self):
        self.model = Sequential()
         
    #@params num_input_nodes Number of input nodes. In this case the amount of
    #modules for feature extraction
    def initializeNN(self, num_input_nodes, num_hidden_nodes=100):
        
        print("Creating model...")
        
        #This is a well-known heuristic where the number of nodes 
        #on the hidden layer is the mean of the input and output layers
        #hidden=np.mean([num_input_nodes,1]).astype(int)
        
        #Input layers
        self.model.add(Dense(num_input_nodes,input_dim=num_input_nodes, init="uniform"))
        #self.model.add(Activation("relu"))
        
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
    def trainNN(self, data, labels, files, num_epochs=100, cut=.5):
        
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
        pearson = []

        train_cvscores = []
        train_f1scores = []
        train_prscores = []
        train_rcscores = []
        train_tnlist = []
        train_tplist = []
        train_fnlist = []
        train_fplist = []

        
        kfold = StratifiedKFold(labels, n_folds=10, shuffle=True, random_state=seed)

        for train, test in kfold:
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
            
            # evaluate the model
            train_scores = self.model.evaluate(train_data, train_labels, verbose=0)
            train_prd = self.model.predict(train_data, verbose=0)
            print("%s: %.2f%%" % (self.model.metrics_names[1], train_scores[1]*100))
            train_cvscores.append(train_scores[1] * 100)
            #tst = np.array([item[0] for item in data[test]])
            #pearson.append(pearsonr(tst.ravel(), labels[test].ravel()))
            train_fn_files = {}
            train_fp_files = {}

            for i,x in enumerate(train_prd):
                if x >= cut:
                    train_prd[i]=1.0
                    if(not(train_labels[i])):
                        train_fp_files[train_files[i]]=None
                else:
                    train_prd[i]=0.0
                    if(train_labels[i]):
                        train_fn_files[train_files[i]]=None

            train_f1scores.append(f1_score(train_labels, train_prd, average="binary"))
            train_prscores.append(precision_score(train_labels, train_prd, average="binary"))
            train_rcscores.append(recall_score(train_labels, train_prd, average="binary"))
            train_tn, train_fp, train_fn, train_tp = confusion_matrix(train_labels, train_prd).ravel()
            train_tnlist.append(train_tn)
            train_tplist.append(train_tp)
            train_fnlist.append(train_fn)
            train_fplist.append(train_fp)

            # evaluate the model
            scores = self.model.evaluate(test_data, test_labels, verbose=0)
            prd = self.model.predict(test_data, verbose=0)
            print("%s: %.2f%%" % (self.model.metrics_names[1], scores[1]*100))
            cvscores.append(scores[1] * 100)
            #tst = np.array([item[0] for item in data[test]])
            #pearson.append(pearsonr(tst.ravel(), labels[test].ravel()))
            fn_files = {}
            fp_files = {}

            for i,x in enumerate(prd):
                if x >= cut:
                    prd[i]=1.0
                    if(not(test_labels[i])):
                        fp_files[test_files[i]]=None
                else:
                    prd[i]=0.0
                    if(test_labels[i]):
                        fn_files[test_files[i]]=None

            f1scores.append(f1_score(test_labels, prd, average="binary"))
            prscores.append(precision_score(test_labels, prd, average="binary"))
            rcscores.append(recall_score(test_labels, prd, average="binary"))
            tn, fp, fn, tp = confusion_matrix(test_labels, prd).ravel()
            tnlist.append(tn)
            tplist.append(tp)
            fnlist.append(fn)
            fplist.append(fp)
            lentest.append(len(test))
            lentrain.append(len(train))
            lentotal.append(len(labels))

        print("-----------Results on training set------------")
        print("Accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(train_cvscores), np.std(train_cvscores)))
        print("F1: %.2f (+/- %.2f)" % (np.mean(train_f1scores), np.std(train_f1scores)))
        print("Precision: %.2f (+/- %.2f)" % (np.mean(train_prscores), np.std(train_prscores)))
        print("Recall: %.2f (+/- %.2f)" % (np.mean(train_rcscores), np.std(train_rcscores)))
        print("TN: %.2f (+/- %.2f)" % (np.mean(train_tnlist), np.std(train_tnlist)))
        print("TP: %.2f (+/- %.2f)" % (np.mean(train_tplist), np.std(train_tplist)))
        print("FN: %.2f (+/- %.2f)" % (np.mean(train_fnlist), np.std(train_fnlist)))
        print("FP: %.2f (+/- %.2f)" % (np.mean(train_fplist), np.std(train_fplist)))
        #print("Pearson correlation: %.2f (+/- %.2f)" % (np.mean(pearson), np.std(pearson)))
        print("TOTAL TEST: %.2f (+/- %.2f)" % (np.mean(lentest), np.std(lentest)))
        print("TOTAL TRAIN: %.2f (+/- %.2f)" % (np.mean(lentrain), np.std(lentrain)))
        print("TOTAL: %.2f (+/- %.2f)" % (np.mean(lentotal), np.std(lentotal)))

        print("-----------Results on testing set------------")
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

        print(fn_files)
        print(fp_files)


        
    #@params test_data a list of numpy arrays. Each array is an input
    #@params test_labels a numpy array with the target data
    def testNN(self, test_data, test_labels):
        print("Testing model...")
        self.model=keras.models.load_model("NN.model")
        (loss,accuracy)=self.model.evaluate(test_data, test_labels, nb_epoch=50, batch_size=100)
        print("loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))
        
        