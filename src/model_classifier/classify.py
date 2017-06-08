# classify_crp.py
import os, sys
from os.path import join, realpath, dirname, isdir, basename
import itertools
# import for pathes for important folders

import csv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axis3d as a3d #@UnresolvedImport
from matplotlib.font_manager import FontProperties

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# visualization help
from prettytable import PrettyTable
import colorsys

from Classifier import Classifier

def compare_parameter_combinations_bin(classifier, comb_params, train_data, train_labels, test_data, test_labels, results_file, save_model=False):
    p_vals = []
    p_names = []
    for k,vals in comb_params.items():
        p_vals.append(vals)
        p_names.append(k)

    combinations = list(itertools.product(*p_vals))
    comb_names = []

    header = ["model", "eval_set", "accuracy", "precision", "recall", "f1", "tn", "fp", "fn", "tp"]
    results_table = PrettyTable(header)

    for i in range(len(combinations)):
        c_name = ""
        for j,n in enumerate(p_names):
            c_name+=(n+"-"+str(combinations[i][j])+"_")
            classifier.set_attribute(n,combinations[i][j])
        c_name = c_name[:-1]
        classifier.re_initialize()
        classifier.train(train_data, train_labels)

        for eval_set in [(train_data,train_labels,"train"), (test_data,test_labels,"test")]:

            preds = classifier.predict_binary(eval_set[0])
            results_row = [c_name,eval_set[2]]
            # add results
            results_row.append(accuracy_score(eval_set[1], preds))
            results_row.append(precision_score(eval_set[1], preds, average="binary"))
            results_row.append(recall_score(eval_set[1], preds, average="binary"))
            results_row.append(f1_score(eval_set[1], preds, average="binary"))
            results_row.extend(confusion_matrix(eval_set[1], preds).ravel())
            # format results
            results_row[2:6] = ["%.3f"%(val,) for val in results_row[2:6]]
            results_row[6:] = ["%d   %.3f"%(val,val/len(eval_set[1])) for val in results_row[6:]]
            results_table.add_row(results_row)
        results_table.add_row(['']*len(header))

        if(save_model):
            if(not(isdir("models"))):
                os.makedirs("models")
            classifier.save_model(join("models",c_name))

    # write results to file
    with open(results_file, 'w') as f:
        f.write(("Parameter Combination Results:").center(80) + "\n\n")
        f.write(("parameter order:").center(80) + "\n")
        for n in p_names:
            f.write((n).center(80) + "\n")
        f.write(results_table.get_string())

def crossvalidate_proba_bin(model, data, labels, kfolds=10, shuffle=True, seed=None):

    # the seed makes sure to get the same random distribution every time
    if(not(seed is None)):
        np.random.seed(seed)
    kfold = StratifiedKFold(n_splits=kfolds, shuffle=shuffle, random_state=seed)

    # count the itrations
    kfold_iter = 0

    train_results = np.ones((len(data),kfolds))*-1
    test_results = np.ones((len(data),kfolds))*-1
    # test_results = np.zeros((len(data),2))

    # evaluate each split
    for train, test in kfold.split(data,labels):

        model.re_initialize()
        # split the data
        train_data = data[train]
        test_data = data[test]
        train_labels = labels[train]
        test_labels = labels[test]

        # train the model
        model.train(train_data, train_labels)

        # predict the training set
        train_probs = model.predict_proba_binary(train_data)[:]
        train_results[train,kfold_iter] = train_probs

        # predict the testing set
        test_probs = model.predict_proba_binary(test_data)
        test_results[test,kfold_iter] = test_probs
        # test_results[test,0] = test_probs
        # test_results[test,1] = kfold_iter
        kfold_iter += 1

    return train_results, test_results

def analyse_crossvalidation_results_bin(results, labels, model_name, thres=[0.25,0.5,0.75], boxplots=True):

    # those are the names of the overall scores comouted during each run
    score_names = ["accuracy", "precision", "recall", "f1", "tn", "fp", "fn", "tp"]
    # evaluate the results considering different thresholds
    test_size = len(labels)/len(results[0])
    for t in thres:
        # create a table
        results_table = PrettyTable(score_names)
        # get the average scores
        average_scores = np.zeros((8,))
        for k in range(0,len(results[0])):
            indices = results[:,k]>=0
            probs = results[indices,k]
            preds = probs>t
            target = labels[indices]

            results_row = []
            results_row.append(accuracy_score(target, preds))
            results_row.append(precision_score(target, preds, average="binary"))
            results_row.append(recall_score(target, preds, average="binary"))
            results_row.append(f1_score(target, preds, average="binary"))
            results_row.extend(confusion_matrix(target, preds).ravel())

            average_scores += np.array(results_row)
            results_row[0:4] = ["%.3f"%(val,) for val in results_row[0:4]]
            results_row[4:] = ["%d   %.3f"%(val,val/test_size) for val in results_row[4:]]
            results_table.add_row(results_row)

        results_table.add_row(['']*len(score_names))
        average_scores /= len(results[0])
        average_scores = list(average_scores)
        average_scores[0:4] = ["%.3f"%(val,) for val in average_scores[0:4]]
        average_scores[4:] = ["%d   %.3f"%(int(val),val/test_size) for val in average_scores[4:]]
        results_table.add_row(average_scores)

        with open(join(model_name +"cross_eval_thres_%.3f.txt"%(t,)), 'w') as f:
            f.write(("Crossvalidation results for threshold %.3f"%(t,)).center(80) + "\n")
            f.write(results_table.get_string())

    if(boxplots):
        all_probs = (results.T).reshape(-1)
        all_labels = np.array(list(labels)*len(results[0]))
        create_boxplot(np.concatenate((all_pros[all_lables==0],all_pros[all_lables==0]),axis=1), ["label:0","label:1"], join(model_name, "probs_per_label.png"))

def analyse_crossvalidation_errors(results, model_name, thres=0.5):
    pass

def generate_boxplots_mia_data(data, header, labels):
    pass

def load_data(data_file, error_feature=False, rep_nan=True, norm=True):
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
    np_features = features.as_matrix(column_names[2:])
    # get classification
    np_classes = np.array(features[column_names[1]].tolist())
    # get the document_ids
    filenames = features[column_names[0]].tolist()
    # cut away doc_id and classification column name
    column_names = column_names[2:]

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
    colors = ["red","black"]
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

    train_csv = args[1]
    test_csv = args[2]

    train_data, train_labels, train_docs, column_names = load_data(train_csv)
    test_data, test_labels, test_docs, column_names = load_data(test_csv)

    # # Create a Logistic Regression Instance
    # from Logistic_Regression import Logistic_Regression
    # kwargs = {"penalty":'l2', "C":1, "fit_intercept":True, "intercept_scaling":1000}
    # lr = Logistic_Regression(**kwargs)

    # # Create a Random Forest Instace
    # from Random_Forest import Random_Forest
    # kwargs = {"n_estimators":10, "criterion":'gini', "max_depth":10, "min_samples_split":2, "min_samples_leaf":1, "min_weight_fraction_leaf":0.0, "max_features":'auto', "max_leaf_nodes":None, "min_impurity_split":1e-07, "bootstrap":True, "oob_score":False, "n_jobs":1, "random_state":None, "verbose":0, "warm_start":False, "class_weight":None}
    # rf = Random_Forest(**kwargs)

    # Create a MLP Instance
    from Keras_Dense_MLP import Keras_Dense_MLP
    layer_params = {"kernel_initializer":"glorot_uniform", "activation":"sigmoid"}
    compile_params = {"loss":"mean_squared_error", "optimizer":"sgd", "metrics":["accuracy"]}
    train_params = {"epochs":100000, "batch_size":64, "verbose":0}
    mlp = Keras_Dense_MLP(neuron_layers=[len(train_data[0]),500,1], layer_params=layer_params, compile_params=compile_params, **train_params)

    # # Do Crossvalidation
    # train_res,test_res = crossvalidate_proba_bin(rf,data=features, labels=classes, kfolds=10, shuffle=True, seed=7)
    #
    # analyse_crossvalidation_results(results=test_res, labels=classes, model_name="RF_test_", thres=[0.5], boxplots=False)
    #
    # analyse_crossvalidation_results(results=train_res, labels=classes, model_name="RF_train_", thres=[0.5], boxplots=False)

    # Compare parameter combinations
    results_file = "MLP_evaluation.txt"
    comb_params = {"neuron_layers":[[len(train_data[0]),64,32,1],[len(train_data[0]),64,1],[len(train_data[0]),64,32,16,1]]}
    compare_parameter_combinations_bin(mlp, comb_params, train_data, train_labels, test_data, test_labels, results_file, True)
