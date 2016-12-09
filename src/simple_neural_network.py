# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 12:09:06 2016

Simple neural net for the prototype

@author: Renato Garita Figueiredo
"""

import sys, os

from os.path import join, isdir
from doc_globals import* 

from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

import numpy as np
import keras
from scipy.stats import pearsonr, mstats

from prettytable import PrettyTable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axis3d as a3d #@UnresolvedImport
from matplotlib.font_manager import FontProperties
import colorsys

class NN:
    
    def __init__(self, num_input_nodes, num_hidden_nodes=100, num_hidden_layers=1, modelname="NN",pretrained_model=None):
        if(not(pretrained_model is None)):
            print("Loading Model")
            self.model = keras.models.load_model(pretrained_model)
        else:

            self.model = Sequential()
            
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

            if(not(isdir(MODEL_PATH))):
                os.mkdir(MODEL_PATH)
            self.model_dir = join(MODEL_PATH,modelname+".model")

            self.model.save(self.model_dir)        
        
        
    #@params data a list of numpy arrays. Each array is an input
    #@params labels a numpy array with the target data

    def train_testNN(self, train_data, train_labels, test_data, test_lables, num_epochs):
        self.model.fit(train_data, train_labels, nb_epoch=num_epochs, verbose=0)
        train_scores = self.model.evaluate(test_data, test_lables, verbose=0)
        print("train_%s: %.2f%%" % (self.model.metrics_names[1], train_scores[1]*100))
        prd = self.model.predict(test_data, verbose=0)
        for p,lab in zip(prd,test_lables):
            print("%.3f\t"%(p,) + str(lab))
        self.model.save("NN.model") 

    def testNN(self, data, labels):

        train_scores = self.model.evaluate(data, labels, verbose=0)
        print("train_%s: %.2f%%" % (self.model.metrics_names[1], train_scores[1]*100))
        prd = self.model.predict(data, verbose=0)
        for p,lab in zip(prd,labels):
            print("%.3f\t"%(p,) + str(lab))


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

        

    def k_fold_crossvalidation(self, data, labels, files, f_names, n_folds, outpath, num_epochs=100, cut=.5):

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
            train_fn_idx = {}
            train_fp_idx = {}
            test_fn_idx = {}
            test_fp_idx = {}

            train_copyright_box = []
            train_non_copyright_box = []
            test_copyright_box = []
            test_non_copyright_box = []

            test_fn_rows = []
            test_fp_rows = []
            test_unsure_rows = []

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
                self.model = keras.models.load_model(self.model_dir)
                # train the model
                self.model.fit(train_data, train_labels, nb_epoch=num_epochs, verbose=0)

                # print(np.mean(self.model.get_weights()[0], axis=1))
                # print(np.std(self.model.get_weights()[0], axis=1))
                # print(np.max(self.model.get_weights()[0], axis=1))
                # sys.exit(1)

                train_row = []
                train_scores = self.model.evaluate(train_data, train_labels, verbose=0)
                print("train_%s: %.2f%%" % (self.model.metrics_names[1], train_scores[1]*100))
                train_row.append(train_scores[1] * 100)

                train_prd = self.model.predict(train_data, verbose=0)
                train_prd_bin = np.zeros((len(train_prd),1))
                for i,x in enumerate(train_prd):
                    x=float(x[0])
                    if x >= cut:
                        train_prd_bin[i]=1
                        if(not(train_labels[i])):
                            if(train[i] in train_fp_idx):
                                train_fp_idx[train[i]].append(x)
                            else:
                                train_fp_idx[train[i]] = [x]
                            train_non_copyright_box.append(x)
                        else:
                            train_copyright_box.append(x)
                    else:
                        train_prd_bin[i]=0
                        if(train_labels[i]):
                            if(train[i] in train_fn_idx):
                                train_fn_idx[train[i]].append(x)
                            else:
                                train_fn_idx[train[i]] = [x]
                            train_copyright_box.append(x)
                        else:
                            train_non_copyright_box.append(x)

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
                        x=float(x[0])
                        test_prd_bin[i]=1.0
                        if(not(test_labels[i])):
                            test_fp_idx[test[i]] = x
                            test_fp_rows.append([files[test[i]],x])
                            test_non_copyright_box.append(x)
                        else:
                            test_copyright_box.append(x)
                        if(x<=0.7):
                            test_unsure_rows.append([files[test[i]],x])

                    else:
                        test_prd_bin[i]=0.0
                        if(test_labels[i]):
                            test_fn_idx[test[i]] = x
                            test_fn_rows.append([files[test[i]],x])
                            test_copyright_box.append(x)
                        else:
                            test_non_copyright_box.append(x)
                        if(x>=0.3):
                            test_unsure_rows.append([files[test[i]],x])

                test_row.append(f1_score(test_labels, test_prd_bin, average="binary"))
                test_row.append(precision_score(test_labels, test_prd_bin, average="binary"))
                test_row.append(recall_score(test_labels, test_prd_bin, average="binary"))
                test_row.extend(confusion_matrix(test_labels, test_prd_bin).ravel())

                test_table.add_row(val_list_to_strings(test_row))
                test_mat[kfold_iter,:] = test_row

                kfold_iter+=1
                self.model.save(self.model_dir)


            test_table.add_row(['']*len(score_names))
            train_table.add_row(['']*len(score_names))

            test_table.add_row(val_list_to_strings(np.mean(test_mat,axis=0)))
            test_table.add_row(val_list_to_strings(np.std(test_mat,axis=0)))
            train_table.add_row(val_list_to_strings(np.mean(train_mat,axis=0)))
            train_table.add_row(val_list_to_strings(np.std(train_mat,axis=0)))

            header = range(0,len(f_names)+2)

            fp_test_table = PrettyTable(header)
            for idx in test_fp_idx.keys():
                row = [files[idx]]
                row.append(test_fp_idx[idx])
                row.extend(data[idx,:])
                fp_test_table.add_row(val_list_to_strings(row))

            fn_test_table = PrettyTable(header)
            for idx in test_fn_idx.keys():
                row = [files[idx]]
                row.append(test_fn_idx[idx])
                row.extend(data[idx,:])
                fn_test_table.add_row(val_list_to_strings(row))

            fp_train_table = PrettyTable(header)
            for idx in train_fp_idx.keys():
                row = [files[idx]]
                row.append(train_fp_idx[idx])
                row.extend(data[idx,:])
                fp_train_table.add_row(val_list_to_strings(row))

            fn_train_table = PrettyTable(header)
            for idx in train_fn_idx.keys():
                row = [files[idx]]
                row.append(train_fn_idx[idx])
                row.extend(data[idx,:])
                fn_train_table.add_row(val_list_to_strings(row))

            text_width_scores = len(test_table.get_string().split("\n")[0])
            text_width_files = len(fn_test_table.get_string().split("\n")[0])


            legend = ["document_id", "prediction"]
            legend.extend(f_names)
            field_names = ""
            for i,name in enumerate(legend):
                field_names += str(i)+":"+"%-20s"%(name,)+"\t"
                if((i+1)%4==0):
                    field_names+="\n"

            with open(join(outpath, "cross_eval.txt"), 'w') as f:
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

            test_fn_rows.sort(key=lambda list: list[1], reverse=False)
            test_fp_rows.sort(key=lambda list: list[1], reverse=True)
            test_unsure_rows.sort(key=lambda list: list[1], reverse=False)

            text_width_files = len("%s\t%.3f"%(test_fn_rows[0][0],test_fn_rows[0][1]))
            with open(join(outpath,"false_pred_list.txt"), 'w') as f:

                f.write("False Classified Documents".center(80))
                f.write("\n")
                f.write("(High percentages mean a high confidence that the document is copyright protected)".center(80))
                f.write("\n")

                f.write("\n\n")
                f.write("False Negatives".center(80))
                f.write("\n")
                f.write("(Classified as not copyright protected but they are)".center(80))
                f.write("\n\n")
                for row in test_fn_rows:
                    f.write(("%s\t%.3f"%(row[0],row[1])).center(80))
                    f.write("\n")

                f.write("\n\n")
                f.write("False Positives".center(80))
                f.write("\n")
                f.write("(Classified copyright protected but they are not)".center(80))
                f.write("\n\n")
                for row in test_fp_rows:
                    f.write(("%s\t%.3f"%(row[0],row[1])).center(80))
                    f.write("\n")

                f.write("\n\n")
                f.write("Unsecure Classifications".center(80))
                f.write("\n")
                f.write("(Classification is not vague)".center(80))
                f.write("\n\n")
                for row in test_unsure_rows:
                    f.write(("%s\t%.3f"%(row[0],row[1])).center(80))
                    f.write("\n")



            test_cb = np.array(test_copyright_box)
            test_ncb = np.array(test_non_copyright_box)
            train_cb = np.array(train_copyright_box)
            train_ncb = np.array(train_non_copyright_box)
            data = [test_cb, test_ncb, train_cb, train_ncb]
            create_boxplot(data, ["prd_test_c", "prd_test_nc", "prd_train_c", "prd_train_nc"], join(outpath,"prediction_boxplot.png"))

    #@params test_data a list of numpy arrays. Each array is an input
    #@params test_labels a numpy array with the target data
    # def testNN(self, test_data, test_labels):
    #     print("Testing model...")
    #     self.model=keras.models.load_model("NN.model")
    #     (loss,accuracy)=self.model.evaluate(test_data, test_labels, nb_epoch=50, batch_size=100)
    #     print("loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))

def create_boxplot(data, collumn_names, filepath):

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

def val_list_to_strings(vals):
    for i,val in enumerate(vals):
        if(type(val)==np.int64 or type(val)==int):
            vals[i] = str(val)
        elif(type(val)==np.float64 or type(val)==float):
            vals[i] = "%.3f"%(val,)
        else:
            vals[i] = val
    return vals
