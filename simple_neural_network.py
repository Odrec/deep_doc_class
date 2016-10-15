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
import numpy as np
import keras
import MetaHandler


class NN:
    
    def __init__(self):
        self.model = Sequential()
         
    #@params num_input_nodes Number of input nodes. In this case the amount of
    #modules for feature extraction
    def initializeNN(self, num_input_nodes):
        
        print("Creating model...")
        
        #This is a well-known heuristic where the number of nodes 
        #on the hidden layer is the mean of the input and output layers
        #hidden=np.mean([num_input_nodes,1]).astype(int)
        
        #Input layers
        self.model.add(Dense(num_input_nodes,input_dim=num_input_nodes, init="uniform"))
        self.model.add(Activation("relu"))
        
        #Hidden layers
        self.model.add(Dense(10, init="uniform"))
        self.model.add(Activation("relu"))
                       
        #Output layer
        #Activation function Sigmoid
        self.model.add(Dense(1, activation="sigmoid"))
        
        #Compile model
        #The loss function is binary_crossentropy since we are dealing with 
        #just two classes. Stochastic gradient descent
        self.model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
        
        self.model.save("NN.model")        
        
        
    #@params train_data a list of numpy arrays. Each array is an input
    #@params train_labels a numpy array with the target data
    def trainNN(self, train_data, train_labels):
        
        seed=7
        np.random.seed(seed)
        
        cvscores = []
        
        kfold = StratifiedKFold(train_labels, n_folds=10, shuffle=True, random_state=seed)

        for train, test in kfold:

            self.model.fit(train_data[train], train_labels[train], nb_epoch=50, batch_size=128)
            
            # evaluate the model
            scores = self.model.evaluate(train_data[test], train_labels[test], verbose=0)
            print("%s: %.2f%%" % (self.model.metrics_names[1], scores[1]*100))
            cvscores.append(scores[1] * 100)
#            gnb = MultinomialNB()
#            gnb.fit(train_data[train],train_labels[train])
#            
#            print(gnb.coef_)
#            print(gnb.score(train_data, train_labels))
#            print(gnb.predict(train_data))
            #MetaHandler.result_html_confusiontable('conf_table.html',train_labels,gnb.predict(train_data))
        
        print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
        
    #@params test_data a list of numpy arrays. Each array is an input
    #@params test_labels a numpy array with the target data
    def testNN(self, test_data, test_labels):
        print("Testing model...")
        self.model=keras.models.load_model("NN.model")
        (loss,accuracy)=self.model.evaluate(test_data, test_labels, nb_epoch=50, batch_size=100)
        print("loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))
        
        