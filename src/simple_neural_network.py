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

        
        kfold = StratifiedKFold(labels, n_folds=10, shuffle=True, random_state=seed)

        for train, test in kfold:
            train_data = data[train]
            test_data = data[test]
            train_labels = labels[train]
            test_lables = labels[test]
            test_files = [files[i] for i in test]

            self.model = keras.models.load_model("NN.model")

            self.model.fit(train_data, train_labels, nb_epoch=num_epochs, verbose=0)
            
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


        
    #@params test_data a list of numpy arrays. Each array is an input
    #@params test_labels a numpy array with the target data
    def testNN(self, test_data, test_labels):
        print("Testing model...")
        self.model=keras.models.load_model("NN.model")
        (loss,accuracy)=self.model.evaluate(test_data, test_labels, nb_epoch=50, batch_size=100)
        print("loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))
        
        